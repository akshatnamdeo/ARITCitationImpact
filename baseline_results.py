import os
import pickle
import numpy as np
import json
from tqdm import tqdm
import torch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from collections import defaultdict

class CitationBaselineEvaluator:
    def __init__(self, data_dir="./arit_data/processed", random_seed=42):
        """Initialize the baseline evaluator with the path to the preprocessed ARIT data."""
        self.data_dir = data_dir
        self.random_seed = random_seed
        self.time_horizons = None
        self.train_states = None
        self.val_states = None
        self.metadata = None
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Create results directory
        self.results_dir = os.path.join(os.path.dirname(data_dir), "baseline_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_data(self):
        """Load the preprocessed ARIT data."""
        print("Loading preprocessed ARIT data...")
        
        # Load train and validation states
        train_path = os.path.join(self.data_dir, "train_states.pkl")
        val_path = os.path.join(self.data_dir, "val_states.pkl")
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise FileNotFoundError(f"Data files not found in {self.data_dir}")
        
        with open(train_path, 'rb') as f:
            self.train_states = pickle.load(f)
        
        with open(val_path, 'rb') as f:
            self.val_states = pickle.load(f)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.time_horizons = self.metadata['time_horizons']
        
        print(f"Loaded {len(self.train_states)} training and {len(self.val_states)} validation states")
        print(f"Time horizons: {self.time_horizons}")
        
        # Get stats on citation counts to understand the data distribution
        train_citations = [state['citation_count'] for state in self.train_states]
        val_citations = [state['citation_count'] for state in self.val_states]
        
        print("\nCitation Count Statistics:")
        print(f"Training set - Min: {min(train_citations)}, Max: {max(train_citations)}, Mean: {np.mean(train_citations):.2f}, Median: {np.median(train_citations)}")
        print(f"Validation set - Min: {min(val_citations)}, Max: {max(val_citations)}, Mean: {np.mean(val_citations):.2f}, Median: {np.median(val_citations)}")
        
        # Split data by citation count
        self.low_citations_train = [s for s in self.train_states if s['citation_count'] < 20]
        self.high_citations_train = [s for s in self.train_states if s['citation_count'] >= 20]
        self.low_citations_val = [s for s in self.val_states if s['citation_count'] < 20]
        self.high_citations_val = [s for s in self.val_states if s['citation_count'] >= 20]
        
        print(f"\nLow-citation papers (<20): {len(self.low_citations_train)} train, {len(self.low_citations_val)} val")
        print(f"High-citation papers (≥20): {len(self.high_citations_train)} train, {len(self.high_citations_val)} val")
        
        return self.train_states, self.val_states
    
    def time_based_split(self, states, test_ratio=0.2):
        """Split data based on time to prevent leakage."""
        # Sort by time index
        sorted_states = sorted(states, key=lambda x: x['time_index'])
        
        # Split based on time index
        split_idx = int(len(sorted_states) * (1 - test_ratio))
        train_states = sorted_states[:split_idx]
        test_states = sorted_states[split_idx:]
        
        return train_states, test_states
        
    def prepare_features(self, states, feature_set='basic'):
        """Extract features from states for training/evaluation."""
        if feature_set == 'basic':
            features = []
            for state in states:
                feat = [
                    state['reference_diversity'],
                    state['field_impact_factor'],
                    state['collaboration_info'],
                    state['time_index']
                ]
                features.append(feat)
            return np.array(features)
        
        elif feature_set == 'with_network':
            features = []
            for state in states:
                feat = [
                    state['reference_diversity'],
                    state['field_impact_factor'],
                    state['collaboration_info'],
                    state['time_index']
                ]
                
                # Add network features if available, but NOT citation counts
                if 'network_data' in state and state['network_data']:
                    net_data = state['network_data']
                    feat.extend([
                        net_data.get('reference_count_actual', 0),
                        net_data.get('internal_reference_count', 0)
                    ])
                else:
                    feat.extend([0, 0])  # Missing network data
                
                features.append(feat)
            return np.array(features)
        
        elif feature_set == 'with_embeddings':
            features = []
            for state in states:
                # FIXED: Removed citation_count from features
                feat = [
                    state['reference_diversity'],
                    state['field_impact_factor'],
                    state['collaboration_info'],
                    state['time_index']
                ]
                
                # Add PCA-reduced embeddings (first 10 components)
                if 'content_embedding' in state:
                    # Take just first 10 dimensions to keep feature count manageable
                    embedding = state['content_embedding'][:10]
                    feat.extend(embedding)
                
                features.append(feat)
            return np.array(features)
        
        else:
            raise ValueError(f"Unknown feature set: {feature_set}")
        
    def prepare_targets(self, states, horizon_idx=None):
        """Extract target values (future citations) from states."""
        if horizon_idx is not None:
            # Return targets for a specific time horizon
            return np.array([state['future_citations'][horizon_idx] for state in states])
        else:
            # Return targets for all time horizons
            return np.array([state['future_citations'] for state in states])
    
    def evaluate_model(self, y_true, y_pred, name="Model"):
        """Calculate evaluation metrics for predictions."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        
        metrics = {
            'name': name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'spearman': spearman_corr
        }
        
        print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, Spearman={spearman_corr:.4f}")
        return metrics
    
    def evaluate_baseline_mean(self, train_states, val_states, name="Mean Baseline"):
        """Evaluate a baseline that predicts the mean citation count."""
        results = []
        
        for h_idx, horizon in enumerate(self.time_horizons):
            # Get target values for this horizon
            y_train = self.prepare_targets(train_states, horizon_idx=h_idx)
            y_val = self.prepare_targets(val_states, horizon_idx=h_idx)
            
            # Predict mean from training set
            mean_citation = np.mean(y_train)
            y_pred = np.full_like(y_val, mean_citation)
            
            # Evaluate
            print(f"\n{name} - {horizon} month horizon:")
            metrics = self.evaluate_model(y_val, y_pred, name=f"{name} ({horizon}m)")
            results.append(metrics)
        
        return results
    
    def evaluate_baseline_regression(self, train_states, val_states, feature_set='basic', model_type='linear', name=None):
        """Evaluate a regression baseline model."""
        if name is None:
            name = f"{model_type.capitalize()} Regression ({feature_set})"
        
        results = []
        
        for h_idx, horizon in enumerate(self.time_horizons):
            # Prepare features and targets
            X_train = self.prepare_features(train_states, feature_set=feature_set)
            y_train = self.prepare_targets(train_states, horizon_idx=h_idx)
            
            X_val = self.prepare_features(val_states, feature_set=feature_set)
            y_val = self.prepare_targets(val_states, horizon_idx=h_idx)
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Initialize model based on type
            if model_type == 'linear':
                model = LinearRegression()
            elif model_type == 'ridge':
                model = Ridge(alpha=1.0)
            elif model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=self.random_seed)
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(n_estimators=100, random_state=self.random_seed)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val_scaled)
            
            # Evaluate
            print(f"\n{name} - {horizon} month horizon:")
            metrics = self.evaluate_model(y_val, y_pred, name=f"{name} ({horizon}m)")
            results.append(metrics)
        
        return results
    
    def evaluate_graph_baseline(self, train_states, val_states, name="Graph-based Baseline"):
        """Evaluate a simple graph-based baseline that uses reference and citation counts."""
        results = []
        
        # Check if we have network data
        has_network = any('network_data' in state and state['network_data'] for state in train_states)
        
        if not has_network:
            print("Network data not available. Skipping graph-based baseline.")
            return []
        
        for h_idx, horizon in enumerate(self.time_horizons):
            # Prepare features that include network data
            X_train = self.prepare_features(train_states, feature_set='with_network')
            y_train = self.prepare_targets(train_states, horizon_idx=h_idx)
            
            X_val = self.prepare_features(val_states, feature_set='with_network')
            y_val = self.prepare_targets(val_states, horizon_idx=h_idx)
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Use gradient boosting for this baseline
            model = GradientBoostingRegressor(n_estimators=100, random_state=self.random_seed)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val_scaled)
            
            # Evaluate
            print(f"\n{name} - {horizon} month horizon:")
            metrics = self.evaluate_model(y_val, y_pred, name=f"{name} ({horizon}m)")
            results.append(metrics)
        
        return results
    
    def evaluate_log_transform_baseline(self, train_states, val_states, name="Log-transform Baseline"):
        """Evaluate a baseline that predicts citations in log space."""
        results = []
        
        for h_idx, horizon in enumerate(self.time_horizons):
            # Prepare features and targets
            X_train = self.prepare_features(train_states, feature_set='basic')
            y_train = self.prepare_targets(train_states, horizon_idx=h_idx)
            
            X_val = self.prepare_features(val_states, feature_set='basic')
            y_val = self.prepare_targets(val_states, horizon_idx=h_idx)
            
            # Log transform the targets (add 1 to handle zeros)
            y_train_log = np.log1p(y_train)
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model in log space
            model = GradientBoostingRegressor(n_estimators=100, random_state=self.random_seed)
            model.fit(X_train_scaled, y_train_log)
            
            # Make predictions and transform back
            y_pred_log = model.predict(X_val_scaled)
            y_pred = np.expm1(y_pred_log)  # Inverse of log1p
            
            # Evaluate
            print(f"\n{name} - {horizon} month horizon:")
            metrics = self.evaluate_model(y_val, y_pred, name=f"{name} ({horizon}m)")
            results.append(metrics)
        
        return results
    
    def evaluate_all_baselines(self):
        """Evaluate all baseline models and compare their performance."""
        print("\n=== Evaluating Baseline Models with Temporal Validation ===\n")
        
        all_results = {}
        
        # Combine train and validation states for proper temporal splitting
        combined_states = self.train_states + self.val_states
        
        # Create time-based splits
        temporal_train, temporal_test = self.time_based_split(combined_states)
        
        # Filter for low and high citation papers
        low_citations_combined = [s for s in combined_states if s['citation_count'] < 20]
        high_citations_combined = [s for s in combined_states if s['citation_count'] >= 20]
        
        low_train, low_test = self.time_based_split(low_citations_combined)
        high_train, high_test = self.time_based_split(high_citations_combined)
        
        print(f"Temporal split created {len(temporal_train)} training and {len(temporal_test)} testing samples")
        print(f"Low-citation temporal split: {len(low_train)} train, {len(low_test)} test")
        print(f"High-citation temporal split: {len(high_train)} train, {len(high_test)} test")
        
        # Evaluate on the full dataset with temporal split
        print("\n--- Full Dataset Evaluation ---")
        all_results['full'] = {
            'mean': self.evaluate_baseline_mean(temporal_train, temporal_test),
            'linear_reg': self.evaluate_baseline_regression(temporal_train, temporal_test, 
                                                        model_type='linear'),
            'ridge_reg': self.evaluate_baseline_regression(temporal_train, temporal_test, 
                                                        model_type='ridge'),
            'rf_reg': self.evaluate_baseline_regression(temporal_train, temporal_test, 
                                                    model_type='random_forest'),
            'gb_reg': self.evaluate_baseline_regression(temporal_train, temporal_test, 
                                                    model_type='gradient_boosting'),
            'graph': self.evaluate_graph_baseline(temporal_train, temporal_test)
        }
        
        # Evaluate on low-citation papers with temporal split
        if len(low_train) > 0 and len(low_test) > 0:
            print("\n--- Low-Citation Papers (<20) Evaluation ---")
            all_results['low'] = {
                'mean': self.evaluate_baseline_mean(low_train, low_test, 
                                                name="Mean Baseline (Low)"),
                'linear_reg': self.evaluate_baseline_regression(low_train, low_test, 
                                                            model_type='linear', name="Linear Regression (Low)"),
                'gb_reg': self.evaluate_baseline_regression(low_train, low_test, 
                                                        model_type='gradient_boosting', name="Gradient Boosting (Low)"),
                'graph': self.evaluate_graph_baseline(low_train, low_test, 
                                                    name="Graph-based (Low)")
            }
        
        # Evaluate on high-citation papers with temporal split
        if len(high_train) > 0 and len(high_test) > 0:
            print("\n--- High-Citation Papers (>=20) Evaluation ---")
            all_results['high'] = {
                'mean': self.evaluate_baseline_mean(high_train, high_test, 
                                                name="Mean Baseline (High)"),
                'linear_reg': self.evaluate_baseline_regression(high_train, high_test, 
                                                            model_type='linear', name="Linear Regression (High)"),
                'gb_reg': self.evaluate_baseline_regression(high_train, high_test, 
                                                        model_type='gradient_boosting', name="Gradient Boosting (High)"),
                'graph': self.evaluate_graph_baseline(high_train, high_test, 
                                                    name="Graph-based (High)")
            }
        
        self.plot_baseline_comparisons(all_results)
        self.save_results(all_results)
        
        return all_results
    
    def plot_baseline_comparisons(self, all_results):
        """Plot comparison of baseline models."""
        datasets = list(all_results.keys())
        metrics = ['mae', 'rmse', 'r2', 'spearman']
        
        # Create a directory for plots
        plots_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # For each time horizon, create comparison plots
        for h_idx, horizon in enumerate(self.time_horizons):
            for metric in metrics:
                plt.figure(figsize=(12, 8))
                
                # Get all model types from the first dataset
                model_types = list(all_results[datasets[0]].keys())
                
                # Prepare data for plotting
                data = []
                labels = []
                
                for dataset in datasets:
                    for model_type in model_types:
                        if model_type in all_results[dataset] and all_results[dataset][model_type]:
                            result = all_results[dataset][model_type][h_idx]
                            data.append(result[metric])
                            labels.append(f"{model_type.replace('_', ' ').title()} ({dataset.title()})")
                
                # Sort from best to worst
                # For MAE and RMSE, lower is better; for R² and Spearman, higher is better
                if metric in ['mae', 'rmse']:
                    sort_indices = np.argsort(data)
                else:
                    sort_indices = np.argsort(-np.array(data))
                
                data = [data[i] for i in sort_indices]
                labels = [labels[i] for i in sort_indices]
                
                # Create the bar chart
                bars = plt.barh(range(len(data)), data, align='center')
                plt.yticks(range(len(data)), labels)
                
                # Add values to bars
                for i, bar in enumerate(bars):
                    plt.text(bar.get_width() + (max(data) * 0.01), 
                             bar.get_y() + bar.get_height()/2, 
                             f"{data[i]:.4f}", 
                             va='center')
                
                plt.title(f"{metric.upper()} Comparison - {horizon} Month Horizon")
                plt.xlabel(metric.upper())
                plt.ylabel("Model")
                plt.tight_layout()
                
                # Save the plot
                plt.savefig(os.path.join(plots_dir, f"{metric}_{horizon}m_comparison.png"))
                plt.close()
        
        print(f"Comparison plots saved to {plots_dir}")
    
    def save_results(self, all_results):
        """Save results to a JSON file."""
        results_file = os.path.join(self.results_dir, "baseline_results.json")
        
        # Convert numpy values to Python native types for JSON serialization
        def convert_results(result_dict):
            if isinstance(result_dict, dict):
                return {k: convert_results(v) for k, v in result_dict.items()}
            elif isinstance(result_dict, list):
                return [convert_results(item) for item in result_dict]
            elif isinstance(result_dict, np.ndarray):
                return result_dict.tolist()
            elif isinstance(result_dict, np.generic):
                return result_dict.item()
            else:
                return result_dict
        
        serializable_results = convert_results(all_results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {results_file}")

    def compare_with_arit(self, arit_results):
        """Compare baseline results with ARIT results."""
        print("\n=== Comparison with Fixed Baselines vs. ARIT ===")
        print("Note: Previous baseline comparisons showed unrealistic results due to data leakage.")
        print("These are the fair comparisons with properly implemented baselines.\n")
        
        # Load our baseline results
        results_file = os.path.join(self.results_dir, "baseline_results.json")
        if not os.path.exists(results_file):
            print("Baseline results not found. Run evaluate_all_baselines() first.")
            return
        
        with open(results_file, 'r') as f:
            baseline_results = json.load(f)
        
        # Create comparison table for low-citation model
        if 'low' in baseline_results and 'low' in arit_results:
            print("\n=== Low-Citation Model Comparison ===")
            self._print_comparison_table(baseline_results['low'], arit_results['low'], "Low-Citation")
        
        # Create comparison table for high-citation model
        if 'high' in baseline_results and 'high' in arit_results:
            print("\n=== High-Citation Model Comparison ===")
            self._print_comparison_table(baseline_results['high'], arit_results['high'], "High-Citation")
    
    def _print_comparison_table(self, baseline_results, arit_result, model_type):
        """Helper method to print comparison table."""
        metrics = ['mae', 'rmse', 'r2', 'spearman']
        
        # Find the best baseline for each metric
        best_baselines = {}
        for metric in metrics:
            best_value = None
            best_model = None
            
            for model_name, results in baseline_results.items():
                if not results:  # Skip empty results
                    continue
                
                # Use the 12-month horizon (last index) for comparison
                result = results[-1][metric]
                
                if best_value is None or (metric in ['mae', 'rmse'] and result < best_value) or (metric in ['r2', 'spearman'] and result > best_value):
                    best_value = result
                    best_model = model_name
            
            best_baselines[metric] = {'model': best_model, 'value': best_value}
        
        # Print comparison table
        print("\nMetric | Best Baseline | Best Baseline Value | ARIT Value | Improvement")
        print("-----|--------------|-------------------|-----------|------------")
        
        for metric in metrics:
            best_baseline = best_baselines[metric]
            baseline_value = best_baseline['value']
            arit_value = arit_result[metric]
            
            # Calculate improvement percentage
            if metric in ['mae', 'rmse']:
                # Lower is better
                improvement = (baseline_value - arit_value) / baseline_value * 100
                improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
            else:
                # Higher is better
                improvement = (arit_value - baseline_value) / max(0.001, baseline_value) * 100
                improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
            
            print(f"{metric.upper()} | {best_baseline['model'].replace('_', ' ').title()} | {baseline_value:.4f} | {arit_value:.4f} | {improvement_str}")

if __name__ == "__main__":
    # Path to the preprocessed ARIT data
    data_dir = "./arit_data/processed"
    
    # Initialize the evaluator and load data
    evaluator = CitationBaselineEvaluator(data_dir=data_dir)
    evaluator.load_data()
    
    # Evaluate all baseline models
    all_results = evaluator.evaluate_all_baselines()
    
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
    
    evaluator.compare_with_arit(arit_results)
    
    print("\nBaseline evaluation complete!")