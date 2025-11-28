# Permafrost Thaw Threatens Alpine Wet Meadow Ecosystems over the Tibetan Plateau

This repository contains the complete code implementation for the research paper "Permafrost Thaw Threatens Alpine Wet Meadow Ecosystems over the Tibetan Plateau".

## 📋 Repository Overview

### 🏔️ Research Focus
- **Permafrost Thaw Dynamics**: Modeling ground temperature and altitude changes
- **Wet Meadow Ecosystems**: Predicting swamp distribution and ecosystem responses
- **Environmental Drivers**: Analyzing climate and vegetation factors using SEM

### 🛠️ Technical Approach
- **Machine Learning**: Ensemble methods (SVM, RF, GBDT, MLP, XGBoost) for spatial prediction
- **Geospatial Analysis**: Raster data processing and prediction mapping
- **Statistical Modeling**: Structural Equation Modeling for causal inference

## 🚀 Quick Start

### Prerequisites

**Python Dependencies:**

```bash
pip install numpy pandas rasterio scikit-learn xgboost joblib
```
**R Dependencies:**

```r
install.packages(c("tidySEM", "lavaan", "semPlot", "readxl"))
```
### Basic Usage

1. **Train Models:**

    ```python
    # Train altitude prediction models
    python scripts/model_training/train_altitude_models.py
    
    # Train ground temperature models
    python scripts/model_training/train_gt_models.py

    # Train swamp prediction models
    python scripts/model_training/train_swamp_models.py
    ```
    
2. **Make Predictions:**
    ```bash
    # Predict altitude distribution
    python scripts/prediction/predict_altitude.py

    # Predict ground temperature
    python scripts/prediction/predict_ground_temp.py

    # Predict swamp distribution
    python scripts/prediction/predict_swamp.py
    ```

3. **Run SEM Analysis:**

    ```r
    # In R console
    source("./sem_analysis.R")
    ```
## 📊 Model Performance

### Machine Learning Metrics
- **MSE (Mean Squared Error):** Used for regression model evaluation
- **Ensemble Performance:** Improved prediction accuracy through model combination

### SEM Fit Indices
- **Chi-square:** Model fit assessment
- **CFI/RMSEA:** Comparative fit and root mean square error approximation
- **GFI/SRMR:** Goodness of fit and standardized root mean square residual

## 🗺️ Output Files

The prediction scripts generate GeoTIFF files with the same spatial reference as input data:

- `output.tiff` - Basic ensemble predictions
- `swamp_pred.tiff` - Normalized swamp distribution predictions (0-1 scale)

