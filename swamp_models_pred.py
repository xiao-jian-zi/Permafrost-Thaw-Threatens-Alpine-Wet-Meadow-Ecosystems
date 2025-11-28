#!/usr/bin/python
# -*- coding: utf-8 -*-

import joblib
import numpy as np
import rasterio
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define TIFF file paths for swamp prediction
tiff_file_paths = {
    'a': "x.tif",  # x coordinate
    'b': "y.tif",  # y coordinate  
    'c': "pre.tif",  # precipitation
    'd': "temp.tif",  # temperature
    'e': "spei.tiff",  # spei index
    'f': "alt.tiff",  # altitude
    'g': "dem.tif",  # digital elevation model
    'h': "fvc.tif",  # fractional vegetation cover
    'i': "gt.tiff",  # ground temperature
}

# Read TIFF files and store in dictionary
tiff_images = {}
for key, path in tiff_file_paths.items():
    with rasterio.open(path) as src:
        tiff_images[key] = src.read(1)  # Read first band

# Ensure all images have the same dimensions
height, width = tiff_images['a'].shape
for img in tiff_images.values():
    assert img.shape == (height, width), "All images must have the same dimensions"

# Build feature vector
X = np.column_stack([tiff_images['a'].flatten(),
                     tiff_images['b'].flatten(),
                     tiff_images['c'].flatten(),
                     tiff_images['d'].flatten(),
                     tiff_images['e'].flatten(),
                     tiff_images['f'].flatten(),
                     tiff_images['g'].flatten(),
                     tiff_images['h'].flatten(),
                     tiff_images['i'].flatten()])

# Normalize features using MinMaxScaler (same as training)
scalerminmax = MinMaxScaler()
X = scalerminmax.fit_transform(X)

feature_names = ['x', 'y', 'pre', 'temp', 'spei', 'alt', 'dem', 'fvc', 'gt']
X = pd.DataFrame(X, columns=feature_names)

# Load trained models
mlp_model = joblib.load("./swap_mlp_model.pkl")
xgboost_model = joblib.load('./swap_xgboost_model.pkl')

# Make predictions
y_mlp_pred = mlp_model.predict(X)
y_xgboost_pred = xgboost_model.predict(X)

# Ensemble prediction (average of MLP and XGBoost)
y_pred = (y_xgboost_pred + y_mlp_pred) / 2

# Normalize predictions to [0, 1] range
scaler = MinMaxScaler(feature_range=(0, 1))
y_pred_scaled = scaler.fit_transform(y_pred.reshape(-1, 1)).flatten()

# Reshape predictions back to image matrix
predicted_image = y_pred_scaled.reshape((height, width))

# Read metadata from first TIFF file
with rasterio.open(tiff_file_paths['a']) as src:
    metadata = src.meta

# Save prediction results as new TIFF file
output_path = "swamp_pred.tiff"
with rasterio.open(output_path, 'w', **metadata) as dst:
    dst.write(predicted_image, 1)