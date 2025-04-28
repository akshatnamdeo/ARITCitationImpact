#!/usr/bin/env python3
import os
import pickle
import json
import numpy as np
import pandas as pd
from collections import defaultdict

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_data():
    processed_dir = "./arit_data/processed"
    train_path = os.path.join(processed_dir, "train_states.pkl")
    val_path = os.path.join(processed_dir, "val_states.pkl")
    metadata_path = os.path.join(processed_dir, "metadata.json")
    citation_network_path = os.path.join(processed_dir, "citation_network.pkl")
    external_papers_path = os.path.join(processed_dir, "external_papers.pkl")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Train or validation states not found in the processed directory.")
    
    train_states = load_pickle(train_path)
    val_states = load_pickle(val_path)
    metadata = load_json(metadata_path) if os.path.exists(metadata_path) else {}
    
    citation_network = load_pickle(citation_network_path) if os.path.exists(citation_network_path) else None
    external_papers = load_pickle(external_papers_path) if os.path.exists(external_papers_path) else None
    
    return train_states, val_states, metadata, citation_network, external_papers

def convert_to_dataframe(train_states, val_states):
    # Mark each state as either train or validation
    for state in train_states:
        state['set'] = 'train'
    for state in val_states:
        state['set'] = 'val'
        
    all_states = train_states + val_states
    df = pd.DataFrame(all_states)
    
    # Expand the list of future citations into separate columns if available
    if 'future_citations' in df.columns and len(df) > 0 and isinstance(df.iloc[0]['future_citations'], list):
        horizons = len(df.iloc[0]['future_citations'])
        for i in range(horizons):
            df[f'future_citations_{i+1}'] = df['future_citations'].apply(lambda x: x[i] if isinstance(x, list) and len(x) > i else np.nan)
        df['future_citations_mean'] = df[[f'future_citations_{i+1}' for i in range(horizons)]].mean(axis=1)
    return df

def print_summary_statistics(df):
    print("=== Summary Statistics ===")
    metrics = ['citation_count', 'reference_diversity', 'field_impact_factor', 'collaboration_info', 'time_index']
    summary_stats = df[metrics].describe()
    print(summary_stats)
    
def print_category_counts(df):
    if 'primary_category' in df.columns:
        print("\n=== Papers per Primary Category ===")
        category_counts = df['primary_category'].value_counts()
        print(category_counts)
        
def print_correlation_matrix(df):
    metrics = ['citation_count', 'reference_diversity', 'field_impact_factor', 'collaboration_info', 'time_index']
    horizon_cols = [col for col in df.columns if col.startswith("future_citations_")]
    all_metrics = metrics + horizon_cols + ['future_citations_mean']
    all_metrics = [col for col in all_metrics if col in df.columns]
    
    if all_metrics:
        corr = df[all_metrics].corr()
        print("\n=== Correlation Matrix ===")
        print(corr)
    else:
        print("No numerical metrics available for correlation matrix.")

def print_time_index_analysis(df):
    if "time_index" in df.columns:
        print("\n=== Time Index Analysis: Average Citation Count per Time Index ===")
        time_groups = df.groupby("time_index")
        avg_citations = time_groups['citation_count'].mean()
        print(avg_citations)

def print_network_analysis(df):
    if "network_data" in df.columns:
        network_ref_counts = []
        network_cit_counts = []
        for idx, row in df.iterrows():
            net_data = row.get("network_data", {})
            if net_data:
                network_ref_counts.append(net_data.get("reference_count_actual", 0))
                network_cit_counts.append(net_data.get("citation_count_actual", 0))
        if network_ref_counts and network_cit_counts:
            print("\n=== Network Data Analysis ===")
            print("Average Network Reference Count: {:.2f}".format(np.mean(network_ref_counts)))
            print("Average Network Citation Count: {:.2f}".format(np.mean(network_cit_counts)))
        else:
            print("No network data found in the dataset.")

def print_train_val_split(df):
    if "set" in df.columns:
        print("\n=== Train/Validation Split Counts ===")
        print(df['set'].value_counts())

def print_citation_ranges(df):
    buckets = [
        ("0-5", 0, 5),
        ("5-10", 5, 10),
        ("10-20", 10, 20),
        ("20-50", 20, 50),
        ("50-100", 50, 100),
        ("100-300", 100, 300),
        ("400-500", 400, 500),
        ("500-1000", 500, 1000),
        ("1000+", 1000, float("inf"))
    ]
    citation_counts = df['citation_count']
    print("\n=== Citation Ranges ===")
    bucket_total = 0
    for label, low, high in buckets:
        if high == float("inf"):
            count = citation_counts[citation_counts >= low].count()
        else:
            count = citation_counts[(citation_counts >= low) & (citation_counts < high)].count()
        print(f"{label}: {count}")
        bucket_total += count
    if bucket_total != len(citation_counts):
        missing = len(citation_counts) - bucket_total
        print(f"Other citation counts (not falling into specified buckets): {missing}")

def main():
    try:
        train_states, val_states, metadata, citation_network, external_papers = load_data()
    except FileNotFoundError as e:
        print("Error loading data:", e)
        return
    
    df = convert_to_dataframe(train_states, val_states)
    print("Loaded dataset with total papers:", df.shape[0])
    if 'primary_category' in df.columns:
        print("Available Categories:", df['primary_category'].unique())
    
    print_summary_statistics(df)
    print_category_counts(df)
    print_correlation_matrix(df)
    print_time_index_analysis(df)
    print_network_analysis(df)
    print_train_val_split(df)
    print_citation_ranges(df)
    
    print("\n=== Metadata Summary ===")
    for key, value in metadata.items():
        print(f"{key}: {value}")
    
    if citation_network:
        print("\nCitation network available with", len(citation_network), "papers.")
    if external_papers:
        print("External papers data available with", len(external_papers), "entries.")
    
    print("\nARIT data analysis complete!")

if __name__ == "__main__":
    main()
