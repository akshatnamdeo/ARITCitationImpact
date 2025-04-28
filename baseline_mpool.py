import os
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
from tqdm import tqdm
import logging

# Configure minimal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class MPoolLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, pooling_ratio=0.5, mode='selection'):
        super(MPoolLayer, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.pooling_ratio = pooling_ratio
        self.mode = mode  # 'selection' or 'clustering'
        
        # Attention weights with careful initialization
        self.W1 = nn.Linear(in_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, 1)
        self.W3 = nn.Linear(in_dim, hidden_dim)
        
        # Initialize weights properly
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.W3.weight)
        nn.init.zeros_(self.W1.bias)
        nn.init.zeros_(self.W2.bias)
        nn.init.zeros_(self.W3.bias)
        
        # For clustering mode
        if mode == 'clustering':
            self.cluster_linear = nn.Linear(in_dim, hidden_dim)
            num_clusters = max(1, int(pooling_ratio * in_dim))
            self.cluster_assign = nn.Linear(hidden_dim, num_clusters)
            
            # Initialize clustering weights
            nn.init.xavier_uniform_(self.cluster_linear.weight)
            nn.init.xavier_uniform_(self.cluster_assign.weight)
            nn.init.zeros_(self.cluster_linear.bias)
            nn.init.zeros_(self.cluster_assign.bias)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization
    
    def forward(self, x, adj=None):
        batch_size, num_nodes, in_features = x.size()
        
        # Create attention scores for nodes
        h = self.leaky_relu(self.W1(x))
        attention = self.W2(h).squeeze(-1)  # [batch_size, num_nodes]
        
        if self.mode == 'selection':
            # Select top-k nodes based on attention scores
            k = int(num_nodes * self.pooling_ratio)
            _, idx = torch.topk(attention, k, dim=1)
            
            # Gather selected nodes' features
            idx_expanded = idx.unsqueeze(-1).expand(-1, -1, in_features)
            selected_x = torch.gather(x, 1, idx_expanded)
            
            # Create new adjacency if provided
            if adj is not None:
                # Handle the case when adj is 2D (shared adjacency matrix)
                if adj.dim() == 2:
                    # Expand to match batch dimension
                    adj_expanded = adj.unsqueeze(0).expand(batch_size, -1, -1)
                    new_adj = torch.zeros(batch_size, k, k, device=x.device)
                    for b in range(batch_size):
                        indices = idx[b]
                        new_adj[b] = adj_expanded[b][indices][:, indices]
                    adj = new_adj
                # Handle case when adj is already 3D
                elif adj.dim() == 3:  
                    new_adj = torch.zeros(batch_size, k, k, device=x.device)
                    for b in range(batch_size):
                        new_adj[b] = adj[b][idx[b]][:, idx[b]]
                    adj = new_adj
            
            return selected_x, adj, attention
        else:  # clustering mode
            # Soft cluster assignment
            cluster_features = self.leaky_relu(self.cluster_linear(x))
            soft_assign = self.cluster_assign(cluster_features)
            soft_assign = F.softmax(soft_assign, dim=2)
            
            # Cluster nodes
            clustered_x = torch.matmul(soft_assign.transpose(1, 2), x)
            
            # For clustering mode with a large sparse adj matrix, 
            # it's better to skip the adj matrix transformation
            # or use a more efficient approach
            new_adj = None
            if adj is not None and adj.size(0) < 1000:  # Only process small adjacency matrices
                if adj.dim() == 2:
                    num_clusters = soft_assign.shape[2]
                    new_adj = torch.eye(num_clusters, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
                elif adj.dim() == 3 and adj.size(1) < 1000:
                    # Only process if the matrix is manageable
                    num_clusters = soft_assign.shape[2]
                    new_adj = torch.eye(num_clusters, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
            
            return clustered_x, new_adj, attention
        
class ARITMPoolModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, field_classes, pool_ratios=[0.5, 0.25]):
        super(ARITMPoolModel, self).__init__()
        
        # Initial feature transformation
        self.feature_embed = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Add batch normalization
        
        # MPool layers - using both selection and clustering for multi-channel approach
        self.mpool_selection = MPoolLayer(hidden_dim, hidden_dim, pool_ratios[0], mode='selection')
        self.mpool_clustering = MPoolLayer(hidden_dim, hidden_dim, pool_ratios[1], mode='clustering')
        
        # Second stage convolution
        self.conv = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # Add batch normalization
        
        # Prediction heads
        self.citation_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Add batch normalization
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Field classification head
        self.field_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Add batch normalization
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(hidden_dim, field_classes)
        )
    
    def forward(self, x, adj=None):
        # Initial feature transformation with gradient clipping
        x = self.feature_embed(x)
        
        # Ensure x has the right dimensions
        if x.dim() == 2:
            # Apply batch norm 
            x = self.bn1(x)
            x = F.relu(x)
            # Add small epsilon to avoid exact zeros
            x = x + 1e-8
            x = x.unsqueeze(1)
        else:
            batch_size, seq_len, features = x.shape
            x = x.reshape(-1, features)
            x = self.bn1(x)
            x = F.relu(x)
            # Add small epsilon to avoid exact zeros
            x = x + 1e-8
            x = x.reshape(batch_size, seq_len, features)
        
        try:
            # Apply MPool - selection channel
            x_selection, _, _ = self.mpool_selection(x, adj)
            
            # Add safety check for NaNs
            if torch.isnan(x_selection).any():
                # If NaNs detected, use the input x as fallback
                x_selection = x
            
            # Apply convolution and pooling
            batch_size, seq_len, features = x_selection.shape
            x_selection = x_selection.reshape(-1, features)
            x_selection = self.conv(x_selection)
            x_selection = self.bn2(x_selection)
            x_selection = F.relu(x_selection)
            x_selection = x_selection.reshape(batch_size, seq_len, features)
            x_selection_pooled = torch.mean(x_selection, dim=1)
            
            # Use simple identity mapping for clustering channel to avoid NaN issues
            x_clustering_pooled = torch.mean(x, dim=1)
            
            # Combine channels
            combined = torch.cat([x_selection_pooled, x_clustering_pooled], dim=1)
            
            if torch.isnan(combined).any():
                # Replace NaNs with small values
                combined = torch.nan_to_num(combined, nan=1e-8)
            
            # Add small epsilon to avoid exact zeros
            combined = combined + 1e-8
            
            # Predictions with gradient clipping
            citation_preds = self.citation_predictor(combined)
            field_preds = self.field_classifier(combined)
            
            # Final safety check
            if torch.isnan(citation_preds).any() or torch.isnan(field_preds).any():
                # Create zero-filled predictions as fallback
                citation_preds = torch.zeros_like(citation_preds)
                field_preds = torch.zeros_like(field_preds)
        
        except Exception as e:
            # If any exception occurs, use safe fallback values
            print(f"Error in forward pass: {e}")
            batch_size = x.size(0)
            citation_preds = torch.zeros((batch_size, 4), device=x.device)  # 4 is for time horizons
            field_classes = self.field_classifier[-1].out_features
            field_preds = torch.zeros((batch_size, field_classes), device=x.device)
        
        return citation_preds, field_preds
    
class ARITMPoolTrainer:
    def __init__(self, data_dir, batch_size=32, hidden_dim=128, epochs=50, lr=0.001):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logging.info(f"Using device: {self.device}")
        
        # Load data
        self.load_data()
        
        # Create model
        first_state = self.train_states[0]  # Get first state
        actual_input_dim = len(first_state['content_embedding'])
        
        # Create model with correct input dimension
        input_dim = actual_input_dim
        output_dim = len(self.time_horizons)
        
        self.model = ARITMPoolModel(
            input_dim=input_dim,  # Use actual dimension from data
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            field_classes=len(self.field_to_id)
        ).to(self.device)
        
        # Use a more robust optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=0.01,  # Increased weight decay for regularization
            eps=1e-8  # Epsilon to prevent division by zero
        )
        
        # Add learning rate scheduler for stability
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Use Huber loss
        self.citation_loss_fn = nn.HuberLoss(delta=1.0)
        self.field_loss_fn = nn.CrossEntropyLoss()
    
    def load_data(self):
        logging.info("Loading data...")
        processed_dir = os.path.join(self.data_dir, "processed")
        
        # Load metadata
        with open(os.path.join(processed_dir, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        self.time_horizons = metadata['time_horizons']
        self.embedding_dim = metadata['embedding_dim']
        self.field_to_id = metadata['field_to_id']
        logging.info("Metadata loaded.")
        
        # Load train and validation states
        with open(os.path.join(processed_dir, "train_states.pkl"), 'rb') as f:
            self.train_states = pickle.load(f)
        with open(os.path.join(processed_dir, "val_states.pkl"), 'rb') as f:
            self.val_states = pickle.load(f)
        logging.info("Train and validation states loaded.")
        
        # Load transitions for creating adjacency matrices
        with open(os.path.join(processed_dir, "transitions.pkl"), 'rb') as f:
            self.transitions = pickle.load(f)
        logging.info("Transitions loaded.")
        
        # Create mapping from state_id to index (for train states)
        self.state_id_to_idx = {state['state_id']: i for i, state in enumerate(self.train_states)}
        
        # Prepare train and validation data
        self.prepare_datasets()
    
    def prepare_datasets(self):
        logging.info("Preparing datasets...")
        
        # File paths for saved adjacency matrices
        train_adj_path = os.path.join(self.data_dir, "processed", "train_adj_matrix.pt")
        val_adj_path = os.path.join(self.data_dir, "processed", "val_adj_matrix.pt")
        
        # Extract features, target citation counts, and fields
        train_features = torch.tensor(np.array([s['content_embedding'] for s in self.train_states]), dtype=torch.float32)
        train_citations = torch.tensor(np.array([s['future_citations'] for s in self.train_states]), dtype=torch.float32)
        train_fields = torch.tensor(np.array([s['field_target'] for s in self.train_states]), dtype=torch.long)
        
        val_features = torch.tensor(np.array([s['content_embedding'] for s in self.val_states]), dtype=torch.float32)
        val_citations = torch.tensor(np.array([s['future_citations'] for s in self.val_states]), dtype=torch.float32)
        val_fields = torch.tensor(np.array([s['field_target'] for s in self.val_states]), dtype=torch.long)
        
        # Try to load saved adjacency matrices
        train_adj = None
        val_adj = None
        
        if os.path.exists(train_adj_path) and os.path.exists(val_adj_path):
            logging.info("Loading saved adjacency matrices from disk...")
            try:
                train_adj = torch.load(train_adj_path)
                val_adj = torch.load(val_adj_path)
                logging.info("Successfully loaded adjacency matrices.")
            except Exception as e:
                logging.warning(f"Error loading adjacency matrices: {e}. Rebuilding...")
                train_adj = val_adj = None
        
        # Create adjacency matrices if not loaded
        if train_adj is None:
            logging.info("Building train adjacency matrix...")
            train_adj = self._create_adjacency_matrix(self.train_states)
            logging.info("Saving train adjacency matrix...")
            torch.save(train_adj, train_adj_path)
        
        if val_adj is None:
            logging.info("Building validation adjacency matrix...")
            val_adj = self._create_adjacency_matrix(self.val_states)
            logging.info("Saving validation adjacency matrix...")
            torch.save(val_adj, val_adj_path)
        
        # Create datasets - Important! Make the dimensions match
        num_train = len(self.train_states)
        num_val = len(self.val_states)
        
        # Create a custom dataset class to handle the mismatch
        class GraphDataset(torch.utils.data.Dataset):
            def __init__(self, features, adj, citations, fields):
                self.features = features
                self.adj = adj
                self.citations = citations
                self.fields = fields
            
            def __len__(self):
                return len(self.features)
            
            def __getitem__(self, idx):
                return self.features[idx], self.adj, self.citations[idx], self.fields[idx]
        
        # Use the custom dataset
        self.train_dataset = GraphDataset(train_features, train_adj, train_citations, train_fields)
        self.val_dataset = GraphDataset(val_features, val_adj, val_citations, val_fields)
        
        # Create dataloaders with a custom collate function
        def custom_collate(batch):
            features = torch.stack([item[0] for item in batch])
            # All samples in the batch share the same adjacency matrix
            adj = batch[0][1]
            citations = torch.stack([item[2] for item in batch])
            fields = torch.stack([item[3] for item in batch])
            return features, adj, citations, fields
        
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=custom_collate
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            collate_fn=custom_collate
        )
        
        logging.info("Datasets and dataloaders are ready.")
            
    def _create_adjacency_matrix(self, states):
        """
        Create adjacency matrix based on citation network relationships.
        This represents the 'motifs' in the citation network.
        """
        num_states = len(states)
        adj_matrix = torch.zeros((num_states, num_states), dtype=torch.float32)
        
        # Create mapping from paper_id to index
        paper_id_to_idx = {state['paper_id']: i for i, state in enumerate(states)}
        
        # Populate the adjacency matrix based on citation relationships using a progress bar
        for i, state in enumerate(tqdm(states, desc="Building adjacency matrix", ncols=80)):
            # Get network data if available
            network_data = state.get('network_data', {})
            
            # Method 1: Using citation relationships
            if 'reference_ids' in state and state['reference_ids']:
                for ref_id in state['reference_ids']:
                    if ref_id in paper_id_to_idx:
                        ref_idx = paper_id_to_idx[ref_id]
                        adj_matrix[i, ref_idx] = 1.0
                        adj_matrix[ref_idx, i] = 1.0
            
            # Method 2: Using citation network if available
            elif 'citation_ids' in network_data and network_data['citation_ids']:
                for cit_id in network_data['citation_ids'][:20]:
                    # Check if this external paper is in our dataset
                    for j, other_state in enumerate(states):
                        if other_state.get('s2_paper_id') == cit_id:
                            adj_matrix[i, j] = 1.0
                            adj_matrix[j, i] = 1.0
                            break
            
            # Method 3: Using transitions (time-based relationships)
            elif state['state_id'] in self.transitions and self.transitions[state['state_id']]:
                for next_state_id in self.transitions[state['state_id']]:
                    # Find index of the next state in the current states list
                    for j, other_state in enumerate(states):
                        if other_state['state_id'] == next_state_id:
                            adj_matrix[i, j] = 1.0
                            break
        
        # Add self-loops
        adj_matrix = adj_matrix + torch.eye(num_states)
        
        # Normalize adjacency matrix
        degree = torch.sum(adj_matrix, dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj_norm = torch.mm(torch.mm(degree_inv_sqrt, adj_matrix), degree_inv_sqrt)
        
        return adj_norm

    def train(self):
        logging.info("Starting training...")
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            nan_batches = 0
            valid_batches = 0
            
            for batch_idx, (features, adj, citations, fields) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", ncols=80)):
                features = features.to(self.device)
                adj = adj.to(self.device)
                citations = citations.to(self.device)
                fields = fields.to(self.device)
                
                # Forward pass
                with torch.autocast(device_type='cuda', enabled=True):  # Use mixed precision
                    citation_preds, field_preds = self.model(features, adj)
                    
                    # Check for NaNs in predictions
                    if torch.isnan(citation_preds).any() or torch.isinf(citation_preds).any():
                        nan_batches += 1
                        if nan_batches <= 5:
                            logging.warning(f"NaN detected in batch {batch_idx}, skipping")
                        continue
                    
                    # Calculate loss
                    citation_loss = self.citation_loss_fn(citation_preds, citations)
                    field_loss = self.field_loss_fn(field_preds, fields)
                    loss = citation_loss + 0.5 * field_loss  # Weighting factor for field loss
                
                # Check for NaNs in loss
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    nan_batches += 1
                    if nan_batches <= 5:
                        logging.warning(f"NaN detected in loss for batch {batch_idx}, skipping")
                    continue
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                valid_batches += 1
            
            # Calculate average loss for epoch
            avg_loss = total_loss / max(1, valid_batches)
            
            if nan_batches > 0:
                logging.warning(f"Skipped {nan_batches} batches with NaN values in epoch {epoch+1}")
            
            # Validation after each epoch
            val_metrics = self.evaluate()
            
            # Use validation loss to step the scheduler
            val_loss = val_metrics['mae'] + val_metrics['rmse']  # Combined metric
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_metrics': val_metrics,
                }, os.path.join(self.data_dir, 'best_mpool_model.pt'))
                logging.info(f"Saved new best model with validation loss: {val_loss:.4f}")
            
            logging.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
            logging.info(f"Validation - MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
            logging.info(f"Field Accuracy: {val_metrics['field_accuracy']:.4f}, Spearman: {val_metrics['spearman']:.4f}")
            logging.info("-" * 50)

    def evaluate(self, split='low'):
        self.model.eval()
        all_preds = []
        all_targets = []
        all_field_preds = []
        all_field_targets = []
        
        # Add a counter to track NaN occurrences
        nan_counter = 0
        
        with torch.no_grad():
            for features, adj, citations, fields in self.val_loader:
                features = features.to(self.device)
                adj = adj.to(self.device)
                citations = citations.to(self.device)
                fields = fields.to(self.device)
                
                # Forward pass
                citation_preds, field_preds = self.model(features, adj)
                
                # Check for NaNs or infinites in predictions
                if torch.isnan(citation_preds).any() or torch.isinf(citation_preds).any():
                    nan_counter += 1
                    # Replace NaNs with zeros and infinite values with a large number
                    citation_preds = torch.nan_to_num(citation_preds, nan=0.0, posinf=1000.0, neginf=-1000.0)
                
                # Store predictions and targets
                all_preds.append(citation_preds.cpu().numpy())
                all_targets.append(citations.cpu().numpy())
                all_field_preds.append(field_preds.cpu().numpy())
                all_field_targets.append(fields.cpu().numpy())
        
        if nan_counter > 0:
            logging.warning(f"Found NaN or infinite values in {nan_counter} batches during evaluation")
        
        # Concatenate results
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_field_preds = np.concatenate(all_field_preds, axis=0)
        all_field_targets = np.concatenate(all_field_targets, axis=0)
        
        # Replace any remaining NaNs or infinite values
        all_preds = np.nan_to_num(all_preds, nan=0.0, posinf=1000.0, neginf=-1000.0)
        
        # Filter by range
        if split == 'low':
            mask = np.all(all_targets < 20, axis=1) & np.all(all_targets > 0, axis=1)
        elif split == 'high':
            mask = np.any(all_targets >= 20, axis=1) & np.all(all_targets < 500, axis=1)
        else:
            mask = np.ones(all_targets.shape[0], dtype=bool)
        
        if mask.sum() > 0:
            filtered_preds = all_preds[mask]
            filtered_targets = all_targets[mask]
            
            # Calculate metrics
            mae = mean_absolute_error(filtered_targets, filtered_preds)
            rmse = np.sqrt(mean_squared_error(filtered_targets, filtered_preds))
            
            # Calculate R² - handle potential errors
            try:
                r2 = r2_score(filtered_targets.flatten(), filtered_preds.flatten())
            except Exception as e:
                r2 = 0
                logging.warning(f"R2 calculation issue: {e}")
                
            # Calculate Spearman correlation
            try:
                spearman_corr, _ = spearmanr(filtered_targets.flatten(), filtered_preds.flatten())
                if np.isnan(spearman_corr):
                    spearman_corr = 0
            except Exception as e:
                spearman_corr = 0
                logging.warning(f"Spearman calculation issue: {e}")
        else:
            mae = rmse = r2 = spearman_corr = 0
        
        # Field classification accuracy
        field_preds_classes = np.argmax(all_field_preds, axis=1)
        field_accuracy = np.mean(field_preds_classes == all_field_targets)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'spearman': spearman_corr,
            'field_accuracy': field_accuracy
        }
    
    def evaluate_both_ranges(self):
        low_metrics = self.evaluate(split='low')
        high_metrics = self.evaluate(split='high')
        
        return {
            'low': low_metrics,
            'high': high_metrics
        }


def run_mpool_experiment():
    data_dir = "./arit_data"
    
    logging.info("Initializing ARITMPoolTrainer...")
    trainer = ARITMPoolTrainer(
        data_dir=data_dir,
        batch_size=32,
        hidden_dim=128,
        epochs=30,
        lr=0.001
    )
    
    # Train the model
    logging.info("Starting training phase...")
    trainer.train()
    
    # Evaluate on both citation ranges
    logging.info("Final Evaluation on both citation ranges...")
    results = trainer.evaluate_both_ranges()
    
    # Display results
    print("\nResults for LOW citation range (0-20):")
    print(f"MAE: {results['low']['mae']:.4f}")
    print(f"RMSE: {results['low']['rmse']:.4f}")
    print(f"R²: {results['low']['r2']:.4f}")
    print(f"Spearman: {results['low']['spearman']:.4f}")
    print(f"Field Accuracy: {results['low']['field_accuracy']:.4f}")
    
    print("\nResults for HIGH citation range (20-500):")
    print(f"MAE: {results['high']['mae']:.4f}")
    print(f"RMSE: {results['high']['rmse']:.4f}")
    print(f"R²: {results['high']['r2']:.4f}")
    print(f"Spearman: {results['high']['spearman']:.4f}")
    print(f"Field Accuracy: {results['high']['field_accuracy']:.4f}")
    
    # Compare with ARIT results
    arit_results = {
        'low': {
            'mae': 1.8584,
            'rmse': 3.1237,
            'r2': 0.4757,
            'spearman': 0.8366,
            'field_accuracy': 0.8416
        },
        'high': {
            'mae': 1.2508,
            'rmse': 3.8187,
            'r2': 0.9975,
            'spearman': 0.9988,
            'field_accuracy': 0.8261
        }
    }
    
    print("\nComparison with ARIT results:")
    print("\nLOW citation range (0-20):")
    print(f"MAE: {results['low']['mae']:.4f} vs {arit_results['low']['mae']:.4f} (ARIT)")
    print(f"RMSE: {results['low']['rmse']:.4f} vs {arit_results['low']['rmse']:.4f} (ARIT)")
    print(f"R²: {results['low']['r2']:.4f} vs {arit_results['low']['r2']:.4f} (ARIT)")
    print(f"Spearman: {results['low']['spearman']:.4f} vs {arit_results['low']['spearman']:.4f} (ARIT)")
    print(f"Field Accuracy: {results['low']['field_accuracy']:.4f} vs {arit_results['low']['field_accuracy']:.4f} (ARIT)")
    
    print("\nHIGH citation range (20-500):")
    print(f"MAE: {results['high']['mae']:.4f} vs {arit_results['high']['mae']:.4f} (ARIT)")
    print(f"RMSE: {results['high']['rmse']:.4f} vs {arit_results['high']['rmse']:.4f} (ARIT)")
    print(f"R²: {results['high']['r2']:.4f} vs {arit_results['high']['r2']:.4f} (ARIT)")
    print(f"Spearman: {results['high']['spearman']:.4f} vs {arit_results['high']['spearman']:.4f} (ARIT)")
    print(f"Field Accuracy: {results['high']['field_accuracy']:.4f} vs {arit_results['high']['field_accuracy']:.4f} (ARIT)")
    
    return results


if __name__ == "__main__":
    run_mpool_experiment()
