import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import joblib

file_path = "DATA_PATH"
data = pd.read_excel(file_path)

X = data.iloc[:, :5]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

svm_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gbdt_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

svm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gbdt_model.fit(X_train, y_train)

y_svm_pred = svm_model.predict(X_test)
y_rf_pred = rf_model.predict(X_test)
y_gbdt_pred = gbdt_model.predict(X_test)

mse = mean_squared_error(y_test, y_svm_pred)
print(f'Mean Squared Error: {mse}')

joblib.dump(svm_model, './alt-svm_model.pkl')
joblib.dump(rf_model, './alt-rf_model.pkl')
joblib.dump(gbdt_model, './alt-gbdt_model.pkl')

loaded_model = joblib.load('./alt-svm_model.pkl')
loaded_y_pred = loaded_model.predict(X_test)
loaded_mse = mean_squared_error(y_test, loaded_y_pred)
print(f'Loaded Model Mean Squared Error: {loaded_mse}')