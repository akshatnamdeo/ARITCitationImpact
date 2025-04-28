import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import time
import os
import pickle
import json
import random
from datetime import datetime
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import re
import bisect
import requests

# Set random seed
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class ARITDataPreparator:
    def __init__(self, data_dir="./arit_data", embedding_model="allenai/scibert_scivocab_uncased", embedding_dim=384, citation_cap=500):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.embedding_model_name = embedding_model
        self.embedding_dim = embedding_dim
        self.citation_cap = citation_cap  # Add citation cap

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        self.categories = [
            "cs.LG", "cs.CV", "cs.AI",
            "cs.CL", "cs.NE", "cs.RO", "cs.HC",
            "physics.comp-ph", "physics.data-an",
            "math.ST", "math.OC",
            "q-bio.QM", "q-fin.ST"
        ]
        self.papers_per_category = 10000
        self.time_horizons = [1, 3, 6, 12]
        self.field_to_id = {}  # Map categories to IDs

        print("Loading sentence embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)

        self.papers = []
        self.embeddings = {}
        self.field_centroids = {}
        self.citation_network = None
        self.external_papers = {}  # Storage for external papers metadata
        
    def fetch_arxiv_papers(self, start_date, end_date, categories=None, max_results_per_query=1000):
        if categories is None:
            categories = self.categories

        all_papers = []
        start_year = int(start_date.split('-')[0])
        end_year = int(end_date.split('-')[0])
        years_to_fetch = end_year - start_year + 1

        for category in categories:
            print(f"Fetching papers for category: {category}")
            papers_per_year = self.papers_per_category // years_to_fetch

            for year in range(start_year, end_year + 1):
                year_start = f"{year}-01-01"
                year_end = f"{year}-12-31"
                start = datetime.strptime(year_start, '%Y-%m-%d').strftime('%Y%m%d')
                end = datetime.strptime(year_end, '%Y-%m-%d').strftime('%Y%m%d')

                year_weight = 0.7 + 0.3 * (year - start_year) / max(1, end_year - start_year)
                target_papers = int(papers_per_year * year_weight * 1.5)

                fetched = 0
                for start_idx in range(0, target_papers, max_results_per_query):
                    query = f'cat:{category} AND (submittedDate:[{start}000000 TO {end}235959] OR lastUpdatedDate:[{start}000000 TO {end}235959])'
                    base_url = "http://export.arxiv.org/api/query?"
                    batch_size = min(max_results_per_query, target_papers - start_idx)

                    if batch_size <= 0:
                        continue

                    params = [
                        ('search_query', query),
                        ('start', start_idx),
                        ('max_results', batch_size),
                        ('sortBy', 'submittedDate'),
                        ('sortOrder', 'descending')
                    ]
                    url = base_url + urllib.parse.urlencode(params)

                    print(f"  Fetching {year} batch {start_idx//max_results_per_query + 1} for {category}, target: {batch_size}")

                    try:
                        response = urllib.request.urlopen(url)
                        data = response.read().decode('utf-8')
                        root = ET.fromstring(data)

                        total_results_elem = root.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults')
                        if total_results_elem is not None:
                            total_available = int(total_results_elem.text)
                            print(f"    Total available: {total_available}")

                        batch_papers = []
                        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                            paper = {
                                'title': entry.find('{http://www.w3.org/2005/Atom}title').text.strip(),
                                'authors': [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')],
                                'published': entry.find('{http://www.w3.org/2005/Atom}published').text,
                                'updated': entry.find('{http://www.w3.org/2005/Atom}updated').text,
                                'summary': entry.find('{http://www.w3.org/2005/Atom}summary').text.strip(),
                                'arxiv_id': entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1],
                                'primary_category': entry.find('{http://arxiv.org/schemas/atom}primary_category').get('term'),
                                'categories': [cat.get('term') for cat in entry.findall('{http://www.w3.org/2005/Atom}category')],
                                'links': [link.get('href') for link in entry.findall('{http://www.w3.org/2005/Atom}link')],
                            }
                            batch_papers.append(paper)

                        all_papers.extend(batch_papers)
                        fetched += len(batch_papers)
                        print(f"    Retrieved {len(batch_papers)} papers, total for {category}: {sum(1 for p in all_papers if p['primary_category'] == category)}")
                        time.sleep(3)

                        if fetched >= target_papers or len(batch_papers) < batch_size:
                            print(f"    Reached target or no more papers for {year}. Moving on.")
                            break

                    except Exception as e:
                        print(f"Error fetching data: {e}")
                        time.sleep(10)
                        continue

                print(f"  Completed {year} for {category}, total papers so far: {len(all_papers)}")

        unique_papers = {paper['arxiv_id']: paper for paper in all_papers}
        self.papers = list(unique_papers.values())
        print(f"Total unique papers retrieved: {len(self.papers)}")
        return self.papers

    def fetch_citation_counts(self, api_key=None, batch_size=300, use_authenticated=True):
        """Fetch citation counts using authenticated or unauthenticated Semantic Scholar API, with progress bar."""
        url = "https://api.semanticscholar.org/graph/v1/paper/batch"
        headers = {"x-api-key": api_key} if use_authenticated and api_key else {}

        wait_time = 1.3 if use_authenticated and api_key else 3.5
        print(f"Fetching citation counts with {'authenticated' if api_key else 'unauthenticated'} Semantic Scholar API...")

        total_batches = (len(self.papers) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(total_batches), desc="Fetching citation counts"):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(self.papers))
            batch_papers = self.papers[start:end]
            arxiv_ids_batch = ["arXiv:" + re.sub(r'v\d+$', '', p['arxiv_id']) for p in batch_papers]
            data = {"ids": arxiv_ids_batch}

            retries = 5
            for attempt in range(retries):
                try:
                    response = requests.post(url, json=data, headers=headers, params={"fields": "citationCount,paperId"})
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', wait_time * (attempt + 1)))
                        print(f"\nRate limit hit on batch {batch_idx + 1}/{total_batches}. Sleeping for {retry_after} seconds...")
                        time.sleep(retry_after)
                        continue
                    elif response.ok:
                        response_data = response.json()

                        if isinstance(response_data, dict) and "data" in response_data:
                            results = response_data["data"]
                        elif isinstance(response_data, list):
                            results = response_data
                        else:
                            raise Exception(f"Unexpected response format: {response_data}")

                        if len(results) != len(batch_papers):
                            print(f"\nWarning: Mismatch in number of papers ({len(batch_papers)}) and results ({len(results)}) in batch {batch_idx + 1}!")

                        for j, result in enumerate(results):
                            if result and isinstance(result, dict):
                                batch_papers[j]['citation_count'] = result.get('citationCount', 0)
                                batch_papers[j]['s2_paper_id'] = result.get('paperId', None)
                            else:
                                batch_papers[j]['citation_count'] = 0
                                batch_papers[j]['s2_paper_id'] = None
                        break  # Successful response, move to next batch
                    else:
                        raise Exception(f"API error in batch {batch_idx + 1}: {response.status_code} - {response.text}")
                except Exception as e:
                    if attempt < retries - 1:
                        backoff_time = wait_time * (attempt + 1)
                        print(f"\nError in batch {batch_idx + 1}: {e}. Retrying in {backoff_time} seconds...")
                        time.sleep(backoff_time)
                    else:
                        print(f"\nError in batch {batch_idx + 1}: {e}. Max retries exceeded. Skipping batch.")
                        for paper in batch_papers:
                            paper['citation_count'] = 0
                            paper['s2_paper_id'] = None
            time.sleep(wait_time)

        print("Citation counts fetched via API.")
        
    def build_citation_network(self, api_key=None, batch_size=300, use_authenticated=True):
        """Build citation network with external paper metadata using Semantic Scholar API."""
        print("Building citation network...")
        url = "https://api.semanticscholar.org/graph/v1/paper/batch"
        headers = {"x-api-key": api_key} if use_authenticated and api_key else {}

        wait_time = 1.3 if use_authenticated and api_key else 3.5
        citation_network = {}
        external_papers = {}
        core_paper_ids = {}

        # First, identify all papers that have s2_paper_id
        for paper in self.papers:
            if 's2_paper_id' in paper and paper['s2_paper_id']:
                arxiv_id = paper['arxiv_id']
                s2_id = paper['s2_paper_id']
                core_paper_ids[s2_id] = arxiv_id
                citation_network[arxiv_id] = {
                    'references': [], 
                    'citations': []
                }

        # Now fetch references and citations in batches
        total_papers = len(core_paper_ids)
        s2_ids = list(core_paper_ids.keys())
        
        print(f"Fetching references for {total_papers} papers...")
        for i in tqdm(range(0, total_papers, batch_size), desc="Fetching references"):
            batch_ids = s2_ids[i:i+batch_size]
            
            # Request references
            fields = "references.paperId,references.title,references.year,references.authors"
            data = {"ids": batch_ids}
            
            retries = 5
            for attempt in range(retries):
                try:
                    response = requests.post(url, json=data, headers=headers, params={"fields": fields})
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', wait_time * (attempt + 1)))
                        print(f"\nRate limit hit on batch {i//batch_size + 1}. Sleeping for {retry_after} seconds...")
                        time.sleep(retry_after)
                        continue
                    elif response.ok:
                        results = response.json()
                        
                        # Process each paper and its references
                        for paper_data in results:
                            if not paper_data or 'paperId' not in paper_data:
                                continue
                                
                            s2_id = paper_data['paperId']
                            arxiv_id = core_paper_ids.get(s2_id)
                            
                            if not arxiv_id:
                                continue
                                
                            # Process references
                            references = paper_data.get('references', [])
                            for ref in references:
                                ref_id = ref.get('paperId')
                                if not ref_id:
                                    continue
                                    
                                # Record reference
                                citation_network[arxiv_id]['references'].append(ref_id)
                                
                                # If reference is external, store its metadata
                                if ref_id not in core_paper_ids and ref_id not in external_papers:
                                    external_papers[ref_id] = {
                                        'title': ref.get('title', ''),
                                        'year': ref.get('year'),
                                        'authors': [author.get('name', '') for author in ref.get('authors', [])],
                                        'is_core': False,
                                        'cited_by': [arxiv_id],
                                        'cites': []
                                    }
                                elif ref_id in external_papers:
                                    if arxiv_id not in external_papers[ref_id]['cited_by']:
                                        external_papers[ref_id]['cited_by'].append(arxiv_id)
                                    
                        break  # Successful response, move to next batch
                    else:
                        raise Exception(f"API error in batch {i//batch_size + 1}: {response.status_code} - {response.text}")
                except Exception as e:
                    if attempt < retries - 1:
                        backoff_time = wait_time * (attempt + 1)
                        print(f"\nError in batch {i//batch_size + 1}: {e}. Retrying in {backoff_time} seconds...")
                        time.sleep(backoff_time)
                    else:
                        print(f"\nError in batch {i//batch_size + 1}: {e}. Max retries exceeded. Skipping batch.")
            
            time.sleep(wait_time)
        # Fetch citations
        print(f"Fetching citations for {total_papers} papers...")
        limit_per_request = 100  # Maximum citations to fetch per paper
        
        for idx, (s2_id, arxiv_id) in enumerate(tqdm(core_paper_ids.items(), desc="Fetching citations")):
            citations_url = f"https://api.semanticscholar.org/graph/v1/paper/{s2_id}/citations"
            params = {
                "fields": "citingPaper.paperId,citingPaper.title,citingPaper.year,citingPaper.authors", 
                "limit": limit_per_request
            }
            
            retries = 3
            for attempt in range(retries):
                try:
                    response = requests.get(citations_url, headers=headers, params=params)
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', wait_time * (attempt + 1)))
                        print(f"\nRate limit hit for paper {idx + 1}/{total_papers}. Sleeping for {retry_after} seconds...")
                        time.sleep(retry_after)
                        continue
                    elif response.ok:
                        citations_data = response.json()
                        
                        # Process citations
                        for citation in citations_data.get('data', []):
                            citing_paper = citation.get('citingPaper', {})
                            citing_id = citing_paper.get('paperId')
                            
                            if not citing_id:
                                continue
                                
                            # Record citation
                            citation_network[arxiv_id]['citations'].append(citing_id)
                            
                            # Store external paper metadata if needed
                            if citing_id not in core_paper_ids and citing_id not in external_papers:
                                external_papers[citing_id] = {
                                    'title': citing_paper.get('title', ''),
                                    'year': citing_paper.get('year'),
                                    'authors': [author.get('name', '') for author in citing_paper.get('authors', [])],
                                    'is_core': False,
                                    'cited_by': [],
                                    'cites': [arxiv_id]
                                }
                            elif citing_id in external_papers:
                                if arxiv_id not in external_papers[citing_id]['cites']:
                                    external_papers[citing_id]['cites'].append(arxiv_id)
                        
                        break  # Successful response, move to next paper
                    else:
                        raise Exception(f"API error for paper {idx + 1}/{total_papers}: {response.status_code} - {response.text}")
                except Exception as e:
                    if attempt < retries - 1:
                        backoff_time = wait_time * (attempt + 1)
                        print(f"\nError for paper {idx + 1}/{total_papers}: {e}. Retrying in {backoff_time} seconds...")
                        time.sleep(backoff_time)
                    else:
                        print(f"\nError for paper {idx + 1}/{total_papers}: {e}. Max retries exceeded. Skipping.")
            
            # Sleep to respect rate limits
            time.sleep(wait_time)
            
            # Save progress every 100 papers
            if (idx + 1) % 100 == 0:
                self._save_citation_network_progress(citation_network, external_papers)
        
        # Mark the papers in our core dataset
        for s2_id, arxiv_id in core_paper_ids.items():
            if s2_id in external_papers:
                external_papers[s2_id]['is_core'] = True
                
        # Save final results
        self.citation_network = citation_network
        self.external_papers = external_papers
        
        # Save to disk
        self._save_citation_network_progress(citation_network, external_papers)
        
        # Update papers with network information
        self._update_papers_with_citation_network()
        
        print(f"Citation network built with {len(citation_network)} core papers and {len(external_papers)} total papers")
        return citation_network, external_papers
        
    def _save_citation_network_progress(self, citation_network, external_papers):
        """Save citation network progress to disk."""
        network_path = os.path.join(self.processed_dir, "citation_network.pkl")
        with open(network_path, 'wb') as f:
            pickle.dump(citation_network, f)
            
        external_path = os.path.join(self.processed_dir, "external_papers.pkl")
        with open(external_path, 'wb') as f:
            pickle.dump(external_papers, f)
            
    def _update_papers_with_citation_network(self):
        """Update paper objects with citation network information."""
        if not self.citation_network:
            print("Citation network not built yet. Skipping update.")
            return
            
        # Create mapping from S2 IDs to our paper objects
        s2_to_paper = {}
        for paper in self.papers:
            if 's2_paper_id' in paper and paper['s2_paper_id']:
                s2_to_paper[paper['s2_paper_id']] = paper
        
        # Update papers with citation network information
        for paper in self.papers:
            arxiv_id = paper['arxiv_id']
            if arxiv_id in self.citation_network:
                # Add reference information
                reference_s2_ids = self.citation_network[arxiv_id]['references']
                paper['outgoing_references'] = reference_s2_ids
                
                # Add citation information
                citation_s2_ids = self.citation_network[arxiv_id]['citations']
                paper['incoming_citations'] = citation_s2_ids
                
                # Calculate network statistics
                paper['network_stats'] = {
                    'reference_count': len(reference_s2_ids),
                    'citation_count': len(citation_s2_ids),
                    'internal_reference_count': sum(1 for ref_id in reference_s2_ids if ref_id in s2_to_paper),
                    'internal_citation_count': sum(1 for cit_id in citation_s2_ids if cit_id in s2_to_paper)
                }

    def _extract_institutions(self, authors):
        institutions = []
        for author in authors:
            matches1 = re.findall(r'\((.*?)\)', author)
            parts = author.split(',')
            matches2 = [part.strip() for part in parts[1:] if part.strip()] if len(parts) > 1 else []
            potential_institutions = matches1 + matches2
            institutions.extend([inst for inst in potential_institutions if len(inst) > 3])
        return institutions

    def preprocess_papers(self):
        print("Preprocessing papers...")
        filtered_papers = []
        for paper in tqdm(self.papers):
            paper['published_dt'] = datetime.strptime(paper['published'], '%Y-%m-%dT%H:%M:%SZ')
            paper['updated_dt'] = datetime.strptime(paper['updated'], '%Y-%m-%dT%H:%M:%SZ')
            paper['institutions'] = self._extract_institutions(paper['authors'])
            paper['collaboration_info'] = min(1.0, len(set(paper['institutions'])) / 3.0) if paper['institutions'] else 0.0
            
            now = datetime.now()
            months_old = max((now.year - paper['published_dt'].year) * 12 + (now.month - paper['published_dt'].month), 1)
            paper['age_months'] = months_old
            paper['time_index'] = months_old // 3
            
            paper['citation_count'] = min(paper.get('citation_count', 0), self.citation_cap)  # Cap here
            
            if paper['citation_count'] == 0:
                continue
            
            active_months = min(months_old, 24)
            monthly_rate = paper['citation_count'] / active_months
            growth_factor = 1.02
            paper['future_citations'] = {}
            last_count = paper['citation_count']
            for horizon in self.time_horizons:
                projected = last_count * (growth_factor ** horizon) + monthly_rate * horizon
                paper['future_citations'][horizon] = min(round(max(last_count, projected)), self.citation_cap)  # Cap here
            
            quality_factors = [
                random.uniform(0.3, 0.7),
                min(1.0, len(paper['authors']) / 5.0),
                min(1.0, len(paper['summary']) / 1000.0),
                0.2 if 'arxiv.org' in ''.join(paper['links']) else 0.0,
            ]
            paper['quality_score'] = sum(quality_factors) / len(quality_factors)
            
            filtered_papers.append(paper)
        
        self.papers = filtered_papers
        print(f"Preprocessing complete. Filtered to {len(self.papers)} papers with non-zero citations.")
                
    def compute_embeddings(self):
        print("Computing paper embeddings...")
        abstracts = [paper['summary'] for paper in self.papers]
        paper_ids = [paper['arxiv_id'] for paper in self.papers]
        batch_size = 32
        embeddings = {}

        for i in tqdm(range(0, len(abstracts), batch_size)):
            batch_abstracts = abstracts[i:i+batch_size]
            batch_ids = paper_ids[i:i+batch_size]
            with torch.no_grad():
                batch_embeddings = self.embedding_model.encode(batch_abstracts)
            for j, paper_id in enumerate(batch_ids):
                embeddings[paper_id] = batch_embeddings[j]

        self.embeddings = embeddings
        for paper in self.papers:
            paper['content_embedding'] = embeddings.get(paper['arxiv_id'], np.zeros(self.embedding_dim))

        print(f"Computed embeddings for {len(embeddings)} papers")

    def compute_field_centroids(self):
        print("Computing field centroids...")
        category_papers = defaultdict(list)
        for paper in self.papers:
            if 'content_embedding' in paper and isinstance(paper['content_embedding'], np.ndarray):
                category_papers[paper['primary_category']].append(paper)

        centroids = {}
        for category, papers in category_papers.items():
            embeddings = np.array([paper['content_embedding'] for paper in papers])
            centroids[category] = np.mean(embeddings, axis=0)
            print(f"Computed centroid for {category} from {len(papers)} papers")

        self.field_centroids = centroids
        for paper in self.papers:
            paper['field_centroid'] = self.field_centroids.get(paper['primary_category'],
                                                               np.mean(list(self.field_centroids.values()), axis=0))

    def compute_reference_diversity_using_network(self):
        """Compute reference diversity using the citation network instead of similarity."""
        print("Computing reference diversity using citation network...")
        
        if not self.citation_network:
            print("Citation network not built yet. Using similarity-based reference diversity instead.")
            return self.compute_reference_diversity()
            
        # Create mapping from S2 IDs to our paper objects
        s2_to_paper = {}
        for paper in self.papers:
            if 's2_paper_id' in paper and paper['s2_paper_id']:
                s2_to_paper[paper['s2_paper_id']] = paper
        
        for paper in tqdm(self.papers, desc="Computing reference diversity"):
            arxiv_id = paper['arxiv_id']
            
            # Skip if paper not in citation network
            if arxiv_id not in self.citation_network:
                paper['reference_diversity'] = 0.0
                paper['reference_count'] = 0
                paper['reference_ids'] = []
                continue
                
            # Get references from citation network
            reference_s2_ids = self.citation_network[arxiv_id]['references']
            
            # Find references that are in our core dataset
            internal_references = []
            for ref_id in reference_s2_ids:
                if ref_id in s2_to_paper:
                    internal_references.append(s2_to_paper[ref_id])
            
            if internal_references:
                # Collect categories from references
                ref_categories = []
                for ref_paper in internal_references:
                    ref_categories.extend(ref_paper.get('categories', []))
                
                # Calculate diversity
                category_counts = Counter(ref_categories)
                total = sum(category_counts.values())
                simpson_index = sum((count/total)**2 for count in category_counts.values()) if total > 0 else 1.0
                
                paper['reference_diversity'] = 1 - simpson_index
                paper['reference_count'] = len(reference_s2_ids)
                paper['reference_ids'] = [ref_paper['arxiv_id'] for ref_paper in internal_references]
            else:
                # If no internal references, use external paper categories if available
                ref_categories = []
                for ref_id in reference_s2_ids[:20]:
                    if ref_id in self.external_papers:
                        ref_categories.append("external")
                
                if ref_categories:
                    category_counts = Counter(ref_categories)
                    total = sum(category_counts.values())
                    simpson_index = sum((count/total)**2 for count in category_counts.values()) if total > 0 else 1.0
                    
                    paper['reference_diversity'] = 1 - simpson_index
                    paper['reference_count'] = len(reference_s2_ids)
                    paper['reference_ids'] = []
                else:
                    paper['reference_diversity'] = 0.0
                    paper['reference_count'] = len(reference_s2_ids)
                    paper['reference_ids'] = []
        
        print("Reference diversity computation using citation network complete!")

    def compute_reference_diversity(self):
        print("Computing reference diversity using similarity...")
        category_papers = defaultdict(list)
        for i, paper in enumerate(self.papers):
            category_papers[paper['primary_category']].append((i, paper))
        
        category_indices = {}
        for cat in category_papers:
            papers = category_papers[cat]
            papers.sort(key=lambda x: x[1]['published_dt'])
            category_indices[cat] = {
                'indices': [p[0] for p in papers],
                'timestamps': [p[1]['published_dt'] for p in papers],
                'papers': [p[1] for p in papers]
            }
        
        for i, paper in enumerate(tqdm(self.papers, desc="Computing reference diversity")):
            primary_cat = paper['primary_category']
            published_date = paper['published_dt']
            
            cat_data = category_indices[primary_cat]
            cutoff_idx = bisect.bisect_left(cat_data['timestamps'], published_date)
            potential_references = [(j, p) for j, p in zip(cat_data['indices'][:cutoff_idx], 
                                                        cat_data['papers'][:cutoff_idx]) 
                                    if j != i]
            
            if len(potential_references) >= 5:
                ref_indices, ref_papers = zip(*potential_references)
                ref_embeddings = np.array([p['content_embedding'] for p in ref_papers])
                paper_embedding = paper['content_embedding'].reshape(1, -1)
                
                similarities = cosine_similarity(paper_embedding, ref_embeddings)[0]
                similarity_pairs = list(zip(ref_indices, similarities))
                similarity_pairs.sort(key=lambda x: x[1], reverse=True)
                
                ref_count = random.randint(5, min(20, len(similarity_pairs)))
                references = similarity_pairs[:ref_count]
                
                ref_categories = []
                for j, _ in references:
                    ref_categories.extend(self.papers[j]['categories'])
                
                category_counts = Counter(ref_categories)
                total = sum(category_counts.values())
                simpson_index = sum((count/total)**2 for count in category_counts.values()) if total > 0 else 1.0
                
                paper['reference_diversity'] = 1 - simpson_index
                paper['reference_count'] = len(references)
                paper['reference_ids'] = [self.papers[j]['arxiv_id'] for j, _ in references]
            else:
                paper['reference_diversity'] = 0.0
                paper['reference_count'] = 0
                paper['reference_ids'] = []
        
        print("Reference diversity computation complete!")

    def compute_field_impact(self):
        print("Computing field impact factors...")
        for paper in tqdm(self.papers):
            primary_cat = paper['primary_category']
            if primary_cat in self.field_centroids and 'content_embedding' in paper:
                sim = cosine_similarity([paper['content_embedding']], [self.field_centroids[primary_cat]])[0][0]
                novelty = 1.0 - sim
                quality = paper['quality_score']
                novelty_factor = 4 * novelty * (1 - novelty)
                paper['field_impact_factor'] = quality * novelty_factor
            else:
                paper['field_impact_factor'] = 0.0

    def prepare_state_dataset(self):
        print("Preparing ARIT state dataset...")
        arit_states = []
        for i, paper in enumerate(self.papers):
            required_fields = ['content_embedding', 'field_centroid', 'reference_diversity',
                            'citation_count', 'field_impact_factor', 'collaboration_info', 'time_index']
            missing_fields = [field for field in required_fields if field not in paper]
            if missing_fields:
                print(f"Skipping paper {paper['arxiv_id']} due to missing fields: {missing_fields}")
                continue
            
            # Map primary_category to field_target
            if paper['primary_category'] not in self.field_to_id:
                self.field_to_id[paper['primary_category']] = len(self.field_to_id)
            
            state = {
                'paper_id': paper['arxiv_id'],
                'state_id': i,
                'content_embedding': paper['content_embedding'],
                'field_centroid': paper['field_centroid'],
                'reference_diversity': paper['reference_diversity'],
                'citation_count': min(paper['citation_count'], self.citation_cap),  # Double-check cap
                'field_impact_factor': paper['field_impact_factor'],
                'collaboration_info': paper['collaboration_info'],
                'time_index': paper['time_index'],
                'primary_category': paper['primary_category'],
                'field_target': self.field_to_id[paper['primary_category']],  # Add field_target
                'title': paper['title'],
                'published_date': paper['published'],
                'future_citations': [min(paper['future_citations'][h], self.citation_cap) for h in self.time_horizons],  # Cap and convert
                'network_data': self._prepare_network_features(paper) if self.citation_network else {}  # Add network features if available
            }
            if i == 0:
                print(f"Sample state: {state}")
            arit_states.append(state)
        
        random.shuffle(arit_states)
        split_idx = int(len(arit_states) * 0.8)
        train_states = arit_states[:split_idx]
        val_states = arit_states[split_idx:]

        print(f"Split into {len(train_states)} training and {len(val_states)} validation states")
        self._print_initial_metrics(train_states, val_states)

        train_path = os.path.join(self.processed_dir, "train_states.pkl")
        val_path = os.path.join(self.processed_dir, "val_states.pkl")
        with open(train_path, 'wb') as f:
            pickle.dump(train_states, f)
        with open(val_path, 'wb') as f:
            pickle.dump(val_states, f)

        metadata = {
            'num_papers': len(self.papers),
            'num_train': len(train_states),
            'num_val': len(val_states),
            'categories': self.categories,
            'time_horizons': self.time_horizons,
            'embedding_model': self.embedding_model_name,
            'embedding_dim': self.embedding_dim,
            'date_created': datetime.now().isoformat(),
            'field_to_id': self.field_to_id,  # Save mapping
            'has_citation_network': self.citation_network is not None
        }
        with open(os.path.join(self.processed_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        return train_states, val_states

    def _prepare_network_features(self, paper):
        """Prepare citation network features for the state dataset."""
        if not self.citation_network or 'arxiv_id' not in paper:
            return {}
            
        arxiv_id = paper['arxiv_id']
        if arxiv_id not in self.citation_network:
            return {}
            
        # Get reference and citation counts
        reference_ids = self.citation_network[arxiv_id]['references']
        citation_ids = self.citation_network[arxiv_id]['citations']
        
        # Calculate network statistics
        network_features = {
            'reference_count_actual': len(reference_ids),
            'citation_count_actual': len(citation_ids),
            'reference_ids': reference_ids[:100],
            'citation_ids': citation_ids[:100],
        }
        
        # Add additional statistics if paper has network_stats
        if 'network_stats' in paper:
            network_features.update(paper['network_stats'])
            
        return network_features

    def _print_initial_metrics(self, train_states, val_states):
        print("\n=== Initial Data Metrics ===")
        print(f"Training states: {len(train_states)}")
        print(f"Validation states: {len(val_states)}")

        for dataset, name in [(train_states, "Train"), (val_states, "Val")]:
            citations = [s['citation_count'] for s in dataset]
            diversity = [s['reference_diversity'] for s in dataset]
            impact = [s['field_impact_factor'] for s in dataset]

            print(f"\n{name} Set:")
            print(f"  Citation Count - Mean: {np.mean(citations):.2f}, Std: {np.std(citations):.2f}, Max: {np.max(citations)}")
            for i, h in enumerate(self.time_horizons):
                h_cits = [s['future_citations'][i] for s in dataset]
                print(f"  {h}-Month Future Citations - Mean: {np.mean(h_cits):.2f}, Std: {np.std(h_cits):.2f}")
            print(f"  Reference Diversity - Mean: {np.mean(diversity):.2f}, Std: {np.std(diversity):.2f}")
            print(f"  Field Impact - Mean: {np.mean(impact):.3f}, Std: {np.std(impact):.3f}")
            
            # Add citation network metrics if available
            if self.citation_network:
                network_papers = [s for s in dataset if s.get('network_data') and 'reference_count_actual' in s.get('network_data', {})]
                if network_papers:
                    ref_counts = [s['network_data']['reference_count_actual'] for s in network_papers]
                    cit_counts = [s['network_data']['citation_count_actual'] for s in network_papers]
                    print(f"  Network Reference Count - Mean: {np.mean(ref_counts):.2f}, Std: {np.std(ref_counts):.2f}")
                    print(f"  Network Citation Count - Mean: {np.mean(cit_counts):.2f}, Std: {np.std(cit_counts):.2f}")
                    print(f"  Network Coverage: {len(network_papers) / len(dataset):.2%} of papers")

    def build_transitions(self, states):
        print("Building state transitions...")
        transitions = {}
        states_by_time = defaultdict(list)
        for state in states:
            states_by_time[state['time_index']].append(state)

        for state in tqdm(states):
            state_id = state['state_id']
            time_idx = state['time_index']
            primary_cat = state['primary_category']
            next_time_idx = time_idx + 1

            if next_time_idx not in states_by_time:
                transitions[state_id] = []
                continue

            candidates = [(s['state_id'], cosine_similarity([state['content_embedding']], [s['content_embedding']])[0][0])
                        for s in states_by_time[next_time_idx] if s['primary_category'] == primary_cat]
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                transitions[state_id] = [cand[0] for cand in candidates[:min(10, len(candidates))]]
            else:
                transitions[state_id] = []

        transitions_path = os.path.join(self.processed_dir, "transitions.pkl")
        with open(transitions_path, 'wb') as f:
            pickle.dump(transitions, f)

        print(f"Built transitions for {len(transitions)} states")
        return transitions

    def generate_statistics(self):
        print("\n=== Final ARIT Dataset Statistics ===")
        category_counts = Counter(paper['primary_category'] for paper in self.papers)
        print("Papers per category:", {cat: count for cat, count in category_counts.items()})
        citations = [paper['citation_count'] for paper in self.papers]
        print(f"Citation mean: {np.mean(citations):.2f}, std: {np.std(citations):.2f}")
        diversities = [paper.get('reference_diversity', 0) for paper in self.papers]
        print(f"Diversity mean: {np.mean(diversities):.2f}")
        impacts = [paper.get('field_impact_factor', 0) for paper in self.papers]
        print(f"Impact mean: {np.mean(impacts):.3f}")
        
        # Add citation network statistics if available
        if self.citation_network:
            network_papers = [p for p in self.papers if p.get('arxiv_id') in self.citation_network]
            if network_papers:
                ref_counts = [len(self.citation_network[p['arxiv_id']]['references']) for p in network_papers]
                cit_counts = [len(self.citation_network[p['arxiv_id']]['citations']) for p in network_papers]
                print(f"Citation network coverage: {len(network_papers) / len(self.papers):.2%} of papers")
                print(f"Average references per paper: {np.mean(ref_counts):.2f}")
                print(f"Average citations per paper: {np.mean(cit_counts):.2f}")
                print(f"Total external papers: {len(self.external_papers)}")

    def save_raw_papers(self):
        raw_papers_path = os.path.join(self.raw_dir, "raw_papers.pkl")
        print(f"Saving {len(self.papers)} raw papers to disk...")
        with open(raw_papers_path, 'wb') as f:
            pickle.dump(self.papers, f)
        print(f"Raw papers saved to {raw_papers_path}")

    def load_raw_papers(self):
        raw_papers_path = os.path.join(self.raw_dir, "raw_papers.pkl")
        if os.path.exists(raw_papers_path):
            print(f"Loading raw papers from {raw_papers_path}...")
            with open(raw_papers_path, 'rb') as f:
                self.papers = pickle.load(f)
            print(f"Loaded {len(self.papers)} raw papers")
            return True
        else:
            print("No saved raw papers found.")
            return False

    def load_citation_network(self):
        """Load citation network and external papers from disk if available."""
        network_path = os.path.join(self.processed_dir, "citation_network.pkl")
        external_path = os.path.join(self.processed_dir, "external_papers.pkl")
        
        if os.path.exists(network_path) and os.path.exists(external_path):
            print("Loading citation network from disk...")
            with open(network_path, 'rb') as f:
                self.citation_network = pickle.load(f)
            with open(external_path, 'rb') as f:
                self.external_papers = pickle.load(f)
            print(f"Loaded citation network with {len(self.citation_network)} papers and {len(self.external_papers)} external papers")
            return True
        else:
            print("No saved citation network found.")
            return False
        
    def run_pipeline(self, start_date, end_date, api_key=None, use_saved_raw=False, use_authenticated=True, build_network=True):
        print("Starting ARIT data pipeline...")
        
        if use_saved_raw and self.load_raw_papers():
            print("Using previously saved raw papers")
        else:
            self.fetch_arxiv_papers(start_date, end_date)
            self.save_raw_papers()
        
        self.fetch_citation_counts(api_key=api_key, use_authenticated=use_authenticated)
        
        # Try to load existing citation network
        network_loaded = False
        if build_network and use_saved_raw:
            network_loaded = self.load_citation_network()
        
        # Build citation network if not loaded
        if build_network and not network_loaded:
            self.build_citation_network(api_key=api_key, use_authenticated=use_authenticated)
        
        self.preprocess_papers()
        self.compute_embeddings()
        self.compute_field_centroids()
        
        # Use network-based reference diversity if available
        if self.citation_network:
            self.compute_reference_diversity_using_network()
        else:
            self.compute_reference_diversity()
        
        self.compute_field_impact()
        train_states, val_states = self.prepare_state_dataset()
        self.build_transitions(train_states)
        self.generate_statistics()

        print("ARIT data pipeline complete!")
        return {
            'train_states': train_states,
            'val_states': val_states,
            'transitions': os.path.join(self.processed_dir, "transitions.pkl"),
            'metadata': os.path.join(self.processed_dir, "metadata.json"),
            'citation_network': os.path.join(self.processed_dir, "citation_network.pkl") if self.citation_network else None,
            'external_papers': os.path.join(self.processed_dir, "external_papers.pkl") if self.external_papers else None
        }

if __name__ == "__main__":
    end_date = '2023-12-31'
    start_date = '2018-01-01'
    api_key = ""
    preparator = ARITDataPreparator(data_dir="./arit_data")
    
    use_saved_raw = True
    use_authenticated = True
    build_network = True
    
    results = preparator.run_pipeline(
        start_date, 
        end_date, 
        api_key=api_key, 
        use_saved_raw=use_saved_raw, 
        use_authenticated=use_authenticated,
        build_network=build_network
    )
    
    print("\nARIT data is ready for model training!")
    print(f"Train states: {len(results['train_states'])} papers")
    print(f"Validation states: {len(results['val_states'])} papers")
    print(f"Data stored in: {preparator.processed_dir}")
    
    if results['citation_network']:
        print(f"Citation network data available at: {results['citation_network']}")
        print(f"External papers data available at: {results['external_papers']}")