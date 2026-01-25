# Model Training Guide

This guide covers training and validating sex classification models in SAGE.

## Overview

SAGE trains elastic net logistic regression models to classify samples as male or female based on gene expression data. The training pipeline includes:

1. **Training Fixtures** - Reproducible datasets for testing
2. **Model Training** - Cross-validated elastic net classifier
3. **Model Validation** - Evaluate against benchmark datasets
4. **Model Persistence** - Version and store trained models
5. **Database Integration** - Track model history and audit trails

## Training Fixtures

Training fixtures are pre-generated datasets for testing without needing live GEO data.

### Available Fixtures

**tiny** (10 samples, 20 features, balanced)
- 5 female, 5 male samples
- Quick unit tests, 5 seconds to train
- Used for quick development cycle

**v1** (100 samples, 100 features, balanced)
- 50 female, 50 male samples
- Standard training dataset, 30 seconds to train
- Recommended for production models

**imbalanced** (60 samples, 100 features, imbalanced)
- 45 male, 15 female samples (75/25 split)
- Tests class imbalance handling
- Important for real-world data

### Using Fixtures in Code

```python
from tests.fixtures.training_fixtures import load_training_fixture, get_fixture_metadata

# Load fixture data
X, y = load_training_fixture("v1")

# Get metadata
metadata = get_fixture_metadata("v1")
print(f"Training {metadata['n_samples']} samples with {metadata['n_features']} features")
```

### Using Fixtures with CLI

```bash
# Train using tiny fixture
poetry run python scripts/train_sex_classifier.py \
    --fixture tiny \
    --alpha 0.5 \
    --l1-ratio 0.5 \
    --cv-folds 5

# Train using v1 fixture
poetry run python scripts/train_sex_classifier.py \
    --fixture v1 \
    --name my_model \
    --version 1.0.0

# Train using imbalanced fixture
poetry run python scripts/train_sex_classifier.py \
    --fixture imbalanced \
    --alpha 0.3 \
    --l1-ratio 0.7
```

## Model Training

### Training Pipeline

```python
from src.sage.model_training import SexClassifierTrainer, ModelConfig
from tests.fixtures.training_fixtures import load_training_fixture

# Load training data
X, y = load_training_fixture("v1")

# Create config
config = ModelConfig(
    alpha=0.5,        # L1/L2 ratio (0.5 = balanced)
    l1_ratio=0.5,     # Regularization strength
    random_state=42,  # For reproducibility
    cv_folds=10,      # 10-fold cross-validation
)

# Train model
trainer = SexClassifierTrainer(config=config)
results = trainer.train(X, y)

# Access results
print(f"Accuracy: {results.accuracy:.4f}")
print(f"F1 Score: {results.f1_score:.4f}")
print(f"AUC: {results.auc_score:.4f}")
print(f"Mean CV Score: {results.mean_cv_score:.4f} ± {results.std_cv_score:.4f}")
```

### Model Configuration

**Hyperparameters:**

- **alpha** (default: 0.5) - L1/L2 ratio for elastic net
  - 0.0 = pure L2 (ridge regression)
  - 0.5 = balanced elastic net
  - 1.0 = pure L1 (lasso)
  - Recommended: 0.5 for balanced regularization

- **l1_ratio** (default: 0.5) - Regularization strength
  - 0.0 = no regularization
  - 0.5 = moderate regularization
  - 1.0 = strong regularization
  - Recommended: 0.5 for general use

- **cv_folds** (default: 10) - Cross-validation folds
  - 5 for faster training
  - 10 for better estimates
  - Higher = more accurate but slower

- **random_state** (default: 42) - Random seed
  - Set for reproducibility
  - Same seed = same results

### Training Results

Results include:

- **accuracy** - Overall accuracy across all CV folds
- **precision** - Weighted precision (importance for FP)
- **recall** - Weighted recall (importance for FN)
- **f1_score** - Weighted F1 (balance of precision/recall)
- **auc_score** - Area under ROC curve
- **cv_fold_scores** - Individual fold accuracies
- **mean_cv_score** - Mean accuracy across folds
- **std_cv_score** - Standard deviation of fold scores

## Model Validation

### Validation Workflow

```python
from src.sage.model_training import SexClassifierTrainer
from src.sage.model_validation import ModelValidator
from tests.fixtures.training_fixtures import load_training_fixture

# Train model
X_train, y_train = load_training_fixture("v1")
trainer = SexClassifierTrainer()
trainer.train(X_train, y_train)

# Load test data
from tests.fixtures.training_fixtures import load_training_fixture
X_test, y_test = load_training_fixture("imbalanced")

# Validate
validator = ModelValidator()
report = validator.validate(
    trainer=trainer,
    X_test=X_test,
    y_test=y_test,
    model_name="test_model"
)

# Access report
print(f"Accuracy: {report.accuracy:.4f}")
print(f"Sensitivity: {report.sensitivity:.4f}")
print(f"Specificity: {report.specificity:.4f}")
print(f"TP: {report.true_positives}, TN: {report.true_negatives}")
print(f"FP: {report.false_positives}, FN: {report.false_negatives}")
```

### Benchmark Comparison

SAGE includes benchmark data from Flynn et al. (2021):

```python
# Compare against microarray benchmark
comparison = validator.compare_benchmark(report, benchmark_name="microarray")

print(f"Model Accuracy: {comparison['model_accuracy']:.4f}")
print(f"Benchmark Accuracy: {comparison['benchmark_accuracy']:.4f}")
print(f"Difference: {comparison['difference']:.4f}")
```

**Available Benchmarks:**
- microarray: 91.7% accuracy (Flynn et al., 2021)
- rnaseq: 88.4% accuracy (Flynn et al., 2021)

### Export Validation Report

```python
validator.export_report(report, "validation_report.json")

# JSON format:
# {
#   "model_name": "sex_classifier_v1",
#   "validation_date": "2026-01-25",
#   "accuracy": 0.92,
#   "precision": 0.91,
#   "recall": 0.93,
#   "f1_score": 0.92,
#   "auc_score": 0.95,
#   "sensitivity": 0.93,
#   "specificity": 0.91,
#   "true_positives": 46,
#   "true_negatives": 46,
#   "false_positives": 4,
#   "false_negatives": 4,
#   "total_samples": 100
# }
```

## Model Persistence

### Saving Models

```python
from src.sage.model_persistence import ModelPersistence

persistence = ModelPersistence()

persistence.save_model(
    trainer=trainer,
    model_path="./models",
    name="sex_classifier",
    version="1.0.0",
    description="Initial model trained on v1 fixture",
    training_samples=100,
    accuracy=results.accuracy,
)

# Creates: models/sex_classifier/1.0.0/
#   ├── model.pkl (trained model)
#   └── metadata.json (version info)
```

### Loading Models

```python
loaded_trainer = persistence.load_model(
    model_path="./models",
    name="sex_classifier",
    version="1.0.0"
)

# Make predictions
predictions = loaded_trainer.predict_proba(X_test)
```

### Version Management

```python
# List all versions
versions = persistence.list_versions("./models", "sex_classifier")
print(versions)  # ['1.0.0', '1.1.0', '2.0.0']

# Get latest version
latest = persistence.get_latest_version("./models", "sex_classifier")
print(latest)  # '2.0.0'

# Delete old version
persistence.delete_model("./models", "sex_classifier", "1.0.0")
```

## Database Integration

Track model versions and validation results in Supabase.

### Register Model Version

```python
from src.sage.model_database import ModelDatabaseManager

db_manager = ModelDatabaseManager()

db_manager.register_model_version(
    name="sex_classifier",
    version="1.0.0",
    model_type="elastic_net_logistic_regression",
    training_date="2026-01-25",
    training_samples=100,
    accuracy=results.accuracy,
    precision=results.precision,
    recall=results.recall,
    f1_score=results.f1_score,
    auc_score=results.auc_score,
)
```

### Record Validation

```python
db_manager.record_validation(
    model_name="sex_classifier",
    model_version="1.0.0",
    validation_date="2026-01-25",
    test_samples=50,
    accuracy=report.accuracy,
    precision=report.precision,
    recall=report.recall,
    f1_score=report.f1_score,
    auc_score=report.auc_score,
    sensitivity=report.sensitivity,
    specificity=report.specificity,
    true_positives=report.true_positives,
    true_negatives=report.true_negatives,
    false_positives=report.false_positives,
    false_negatives=report.false_negatives,
    benchmark_name="microarray",
    benchmark_accuracy=0.917,
)
```

### Query Model History

```python
# Get all versions
versions = db_manager.get_model_versions("sex_classifier")

# Get active version
active = db_manager.get_active_model("sex_classifier")

# Get best performing version
best = db_manager.get_best_performing_model("sex_classifier")

# Get audit trail
trail = db_manager.get_audit_trail("sex_classifier")

# Set active version
db_manager.set_active_version("sex_classifier", "2.0.0")

# Export report
report = db_manager.export_model_report("sex_classifier")
```

## CLI Commands

### Train Model

```bash
poetry run python scripts/train_sex_classifier.py \
    --fixture v1 \
    --name sex_classifier \
    --version 1.0.0 \
    --alpha 0.5 \
    --l1-ratio 0.5 \
    --cv-folds 10 \
    --output ./models
```

### Validate Model

```bash
poetry run python scripts/validate_model.py \
    --model sex_classifier \
    --version 1.0.0 \
    --test-data test_data.csv \
    --benchmark microarray \
    --output validation_report.json
```

### Check Model Status

```bash
# List all models
poetry run python scripts/model_status.py --list

# Show specific model details
poetry run python scripts/model_status.py \
    --model sex_classifier \
    --version 1.0.0

# Show best version
poetry run python scripts/model_status.py \
    --model sex_classifier \
    --best
```

## Best Practices

1. **Use Fixtures for Testing**
   - Fixtures ensure reproducible results
   - Use tiny fixture for quick development
   - Use v1 for production models

2. **Cross-Validation**
   - Always use stratified k-fold (default)
   - Use 10 folds for balanced data
   - Use 5 folds for imbalanced data

3. **Hyperparameter Tuning**
   - Start with defaults (alpha=0.5, l1_ratio=0.5)
   - Test alpha values: 0.3, 0.5, 0.7
   - Test l1_ratio values: 0.5, 0.7, 1.0
   - Use small fixtures (tiny) for fast iteration

4. **Model Naming**
   - Use descriptive names: sex_classifier, microarray_classifier
   - Use semantic versioning: 1.0.0, 2.0.0
   - Increment patch for bug fixes
   - Increment minor for improvements
   - Increment major for architectural changes

5. **Validation**
   - Always validate on separate test set
   - Compare against Flynn et al. benchmarks
   - Document validation date and methodology
   - Export reports for audit trail

6. **Database**
   - Register all models in database
   - Record all validation runs
   - Keep audit trail for reproducibility
   - Set active version explicitly

## Troubleshooting

### Model Not Training
- Check fixture exists and is readable
- Verify X and y have matching dimensions
- Ensure y contains only 0 and 1

### Poor Validation Results
- Check test set doesn't contain training data
- Verify test set has similar distribution to training set
- Try different hyperparameters
- Check for class imbalance

### Database Connection Issues
- Verify Supabase credentials in environment
- Check network connectivity
- Try health_check() method to diagnose

### Model Predictions Wrong
- Verify feature count matches training
- Check feature scaling and normalization
- Ensure same preprocessing as training
- Verify model was trained (not just initialized)

## References

- Flynn, E. et al. (2021). "Sex-specific gene expression differences in human blood." BMC Bioinformatics.
- SAGE Dashboard: https://github.com/lfaller/sage-dashboard
- GEO Database: https://www.ncbi.nlm.nih.gov/geo/
