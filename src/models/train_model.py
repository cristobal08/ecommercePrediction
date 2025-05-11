# src/models/train_model.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn

class ModelTrainer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
    
    def train(self, X, y):
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Preprocesamiento
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Entrenamiento con MLflow
        mlflow.set_experiment("sales_prediction")
        
        with mlflow.start_run():
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X_train_processed, y_train)
            
            # Métricas
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            y_pred = model.predict(X_test_processed)
            
            mlflow.log_metric("MAE", mean_absolute_error(y_test, y_pred))
            mlflow.log_metric("MSE", mean_squared_error(y_test, y_pred))
            mlflow.log_metric("R2", r2_score(y_test, y_pred))
            
            mlflow.sklearn.log_model(model, "model")
        
        return model