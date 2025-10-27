# Bangla News Classification Notebook - Structure Guide

## Overview

This notebook implements a comprehensive pipeline for binary text classification of Bangla news articles (Political vs Non-Political) using BanglaBERT.

## Notebook Structure

### **Section 1: Setup & Installation**
- Install required dependencies (transformers, datasets, evaluate, etc.)
- Mount Google Drive
- Import all necessary libraries
- Set random seeds for reproducibility

### **Section 2: Data Loading & EDA**
- Load CSV dataset
- Display basic statistics
- **Class Distribution Analysis**: Visualize label balance with bar plots and pie charts
- **Text Length Analysis**: Character and word count distributions by class

### **Section 3: Data Pre-processing & Tokenization**
- Configuration and paths setup
- Train/Validation/Test split (stratified)
- Tokenization using BanglaBERT tokenizer
- Save tokenized datasets

### **Section 4: Model Training & Hyperparameter Experiments**
- Load tokenized datasets
- Define WeightedTrainer class for handling class imbalance
- **5 Pre-configured Hyperparameter Settings**:
  - `baseline`: Standard configuration for quick testing
  - `higher_lr`: Recommended balanced configuration
  - `larger_batch`: For systems with more GPU memory
  - `aggressive_regularization`: Prevent overfitting
  - `fast_convergence`: Experimental high learning rate
- **Reusable Training Function**: Train multiple configurations easily
- Automated experiment management

### **Section 5: Results Comparison & Analysis**
- Compare all experiment results side-by-side
- Identify best performing configuration
- Visualization of metrics across experiments
- Validation vs Test performance comparison

### **Section 6: Detailed Validation & Error Analysis**
- Load best model from experiments
- Define prediction function
- Test on sample articles
- Batch prediction examples
- Error analysis tools

### **Section 7: Visualization & Performance Analysis**
- Detailed metrics visualization
- Confusion matrix
- Per-class performance metrics
- Training history plots
- Loss curves and accuracy trends

## Key Features

### 1. **Hyperparameter Experimentation Framework**
```python
# Easily toggle experiments on/off
configs_to_train = {
    'baseline': False,
    'higher_lr': True,  # ← Enable this one
    'larger_batch': False,
    ...
}
```

### 2. **Automatic Results Comparison**
- All experiments are automatically tracked
- Side-by-side metric comparison
- Visual comparison charts
- Best model identification

### 3. **Comprehensive EDA**
- Class distribution analysis
- Text length statistics
- Visual exploration of data characteristics

### 4. **Reusable Components**
- `train_model()`: Train with any hyperparameter configuration
- `predict_article()`: Make predictions on new text
- `compute_metrics()`: Consistent metric calculation
- `WeightedTrainer`: Handle class imbalance

## Hyperparameters to Experiment With

| Parameter | Recommended Range | Default |
|-----------|------------------|---------|
| Learning Rate | 1e-5 to 1e-4 | 3e-5 |
| Batch Size | 4, 8, 16, 32 | 8 |
| Epochs | 3 to 10 | 5 |
| Weight Decay | 0.0 to 0.1 | 0.01 |
| Scheduler | linear, cosine, polynomial | linear |
| Warmup Ratio | 0.0 to 0.2 | 0.0 |
| Gradient Accumulation | 1, 2, 4, 8 | 2 |

## Metrics Tracked

- **Accuracy**: Overall correctness
- **F1 Score (Macro)**: Balanced performance across classes
- **Precision (Macro)**: Positive prediction accuracy
- **Recall (Macro)**: Coverage of positive cases
- Per-class metrics for detailed analysis

## Quick Start Guide

### Step 1: Run Setup
Execute cells in Section 1 to install dependencies and import libraries.

### Step 2: Explore Data
Run Section 2 to understand your dataset characteristics.

### Step 3: Preprocess
Execute Section 3 to tokenize and prepare data.

### Step 4: Configure & Train
1. Review hyperparameter configurations in Section 4
2. Enable desired experiments in `configs_to_train`
3. Run training cell
4. Wait for training to complete

### Step 5: Compare Results
Run Section 5 to see comparison of all experiments.

### Step 6: Analyze Best Model
Use Section 6 for detailed analysis of the best performing model.

### Step 7: Visualize
Run Section 7 for comprehensive visualizations.

## Adding Custom Configurations

To add your own hyperparameter configuration:

```python
hyperparameter_configs['my_custom_config'] = {
    'learning_rate': 4e-5,
    'per_device_train_batch_size': 12,
    'gradient_accumulation_steps': 3,
    'num_train_epochs': 6,
    'weight_decay': 0.02,
    'lr_scheduler_type': 'cosine',
    'warmup_ratio': 0.1,
    'description': 'My custom configuration'
}

# Enable it
configs_to_train['my_custom_config'] = True
```

## Output Structure

```
experiments/
├── baseline/
│   ├── checkpoint-XXX/
│   ├── logs/
│   └── config.json
├── higher_lr/
│   ├── checkpoint-XXX/
│   ├── logs/
│   └── config.json
└── ...
```

## Recommended Practices

1. **Start with baseline**: Run baseline first to establish benchmark
2. **One variable at a time**: When creating custom configs, change one hyperparameter at a time
3. **Monitor validation metrics**: Watch for overfitting (val loss increasing while train loss decreases)
4. **Use early stopping**: Prevents wasted computation and overfitting
5. **Save results**: All experiment results are stored in `all_results` dictionary
6. **Compare fairly**: All experiments use same data splits and random seed

## Troubleshooting

### Out of Memory Error
- Reduce `per_device_train_batch_size`
- Reduce `MAX_LENGTH` in preprocessing
- Increase `gradient_accumulation_steps`

### Training Too Slow
- Increase `per_device_train_batch_size`
- Reduce `num_train_epochs`
- Enable `fp16=True` (already default)

### Poor Performance
- Try higher learning rate
- Add warmup (`warmup_ratio > 0`)
- Use cosine scheduler
- Increase epochs
- Check class weights are being applied

## Notes

- All experiments use the same train/val/test split for fair comparison
- Class weights are automatically computed to handle imbalance
- Early stopping prevents overfitting
- Results are automatically saved to Google Drive

## Model Details

- **Base Model**: csebuetnlp/banglabert
- **Task**: Binary Text Classification
- **Classes**: 0 (Non-Political), 1 (Political)
- **Language**: Bangla
- **Max Sequence Length**: 512 tokens

---

