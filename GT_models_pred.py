import joblib
import numpy as np
import rasterio
import pandas as pd

tiff_file_paths = {
    'a': "TEMP.tif",
    'b': "SPEI.tif", 
    'c': "PRE.tif",
    'd': "FVC.tif",
    'e': "DEM.tif",
}

tiff_images = {}
for key, path in tiff_file_paths.items():
    with rasterio.open(path) as src:
        tiff_images[key] = src.read(1)

height, width = tiff_images['a'].shape
for img in tiff_images.values():
    assert img.shape == (height, width), "All images must have the same dimensions"

X = np.column_stack([tiff_images['a'].flatten(),
                     tiff_images['b'].flatten(),
                     tiff_images['c'].flatten(),
                     tiff_images['d'].flatten(),
                     tiff_images['e'].flatten()])

feature_names = ['tmp', 'spei', 'pre', 'fvc', 'dem']
X = pd.DataFrame(X, columns=feature_names)

svm_model = joblib.load('./gt-svm_model.pkl')
rf_model = joblib.load('./gt-rf_model.pkl')
gbdt_model = joblib.load('./gt-gbdt_model.pkl')

y_svm_pred = svm_model.predict(X)
y_rf_pred = rf_model.predict(X)
y_gbdt_pred = gbdt_model.predict(X)

y_pred = (y_svm_pred + y_rf_pred + y_gbdt_pred) / 3

predicted_image = y_pred.reshape((height, width))

with rasterio.open(tiff_file_paths['a']) as src:
    metadata = src.meta

output_path = "output.tiff"
with rasterio.open(output_path, 'w', **metadata) as dst:
    dst.write(predicted_image, 1)