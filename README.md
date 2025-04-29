# ARIT: Adaptive Research Impact Transformer

ARIT (Adaptive Research Impact Transformer) is a novel framework for citation prediction and strategic research positioning optimization that models academic papers as strategic agents within a competitive citation landscape.

## Overview

The academic citation landscape follows a skewed distribution where papers strategically compete for attention and impact. ARIT addresses this challenge through:

1. **Dual-Model Architecture**: Specialized models for low-citation (≤20) and high-citation (>20) papers
2. **Network-Centric Analysis**: Citation graphs with graph attention networks to understand paper positioning
3. **Strategic Optimization**: Reinforcement learning approach for optimal research positioning
4. **Temporal Dynamics Modeling**: Capturing citation pattern evolution over time

## Dataset and Paper

- Pre-processed datasets are available on Google Drive: [ARIT Dataset](https://drive.google.com/drive/folders/1L4pP4QHl-79lReSQph9Rp3JVb0boHhlB?usp=sharing)
- Read our research paper: [ARIT: Adaptive Research Impact Transformer with Graph Attention Networks for Strategic Citation Optimization](https://drive.google.com/file/d/1MDYujsVa7goU_UkLb5vz7BrAXbeJRIgZ/view?usp=sharing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/akshatnamdeo/ARITCitationImpact.git
cd ARIT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

To fetch and preprocess the dataset, first update the API key in the script:

```python
# In fetch_and_preprocess.py, modify the API key line:
api_key = "YOUR_SEMANTIC_SCHOLAR_API_KEY"
```

Then run the preprocessing script:

```bash
python fetch_and_preprocess.py
```

### Dataset Analysis

To analyze the preprocessed dataset:

```bash
python analyze_dataset.py --data_dir ./arit_data/processed
```

### Model Training

The `start_training.py` script supports multiple training configurations and phases through command line flags:

```bash
python start_training.py [FLAGS] [--folder FOLDER_NAME]
```

Available flags:

**Phase flags** (control which training phases to run):
- `--pretrain-high`: Run pretraining for high-citation model only
- `--pretrain-low`: Run pretraining for low-citation model only
- `--pretrain-both`: Run pretraining for both models
- `--pretrain-moe`: Run pretraining for Mixture of Experts model
- `--rl-high`: Run reinforcement learning for high-citation model only
- `--rl-low`: Run reinforcement learning for low-citation model only
- `--rl-both`: Run reinforcement learning for both models
- `--rl-moe`: Run reinforcement learning for Mixture of Experts model
- `--validate-high`: Run validation for high-citation model only
- `--validate-low`: Run validation for low-citation model only
- `--validate-both`: Run validation for both models
- `--validate-moe`: Run validation for Mixture of Experts model
- `--run-all-high`: Run all phases with high-citation model only
- `--run-all-low`: Run all phases with low-citation model only
- `--run-all-both`: Run all phases with both models
- `--run-all-moe`: Run all phases with Mixture of Experts model

**Folder option**:
- `--folder FOLDER_NAME`: Specify a folder name from which to load/save models

Examples:
```bash
# Run all phases with both models
python start_training.py --run-all-both --folder my_results

# Run only pretraining for the high-citation model
python start_training.py --pretrain-high

# Run RL training for models in an existing folder
python start_training.py --folder results_20240428_120000 --rl-both

# Validate existing models
python start_training.py --folder results_20240428_120000 --validate-both
```

### Running Analysis

For final model analysis and visualization:

```bash
python arit_analysis.py --model_path ./model_outputs/best_model.pt --data_dir ./arit_data/processed --output_dir ./analysis_results
```

## Repository Structure

```
├── arit_data/                      # Data directory
│   ├── processed/                  # Processed dataset (download from Google Drive)
│   │   ├── citation_network.pkl    # Citation network data
│   │   ├── external_papers.pkl     # External paper metadata
│   │   ├── metadata.json           # Dataset metadata
│   │   ├── train_adj_matrix.pt     # Training adjacency matrix
│   │   ├── train_states.pkl        # Training states
│   │   ├── transitions.pkl         # State transitions
│   │   ├── val_adj_matrix.pt       # Validation adjacency matrix
│   │   └── val_states.pkl          # Validation states
│   ├── raw/                        # Raw data
│   │   └── raw_papers.pkl          # Original paper data
│   └── final_results/              # Results of model analysis
├── arit_attention.py               # Attention mechanisms implementation
├── arit_citations.py               # Citation prediction heads and loss functions
├── arit_config.py                  # Configuration settings
├── arit_environment.py             # Reinforcement learning environment
├── arit_evaluation.py              # Evaluation metrics
├── arit_model.py                   # Main model architecture
├── arit_training.py                # Training procedures
├── arit_types.py                   # Shared type definitions
├── arit_analysis.py                # Analysis code
├── analyze_dataset.py              # Dataset analysis script
├── baseline_mpool.py               # MPool baseline implementation
├── baseline_results.py             # Other baseline models
├── commands.txt                    # Example commands for training
├── fetch_and_preprocess.py         # Data fetching and preprocessing
├── start_training.py               # Main training script
└── README.md                       # This file
```

## License

MIT License

Copyright (c) 2025 Akshat Namdeo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
