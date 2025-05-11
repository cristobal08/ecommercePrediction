# src/data/data_preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
    
    def preprocess(self, df):
        # Manejo de valores nulos
        df.fillna(method='ffill', inplace=True)
        
        # Ingeniería de características
        df['total_revenue'] = df['Cantidad'] * df['Precio']
        df['dia_semana'] = pd.to_datetime(df['Fecha']).dt.day_name()
        
        # Preparación para modelo
        numeric_features = ['Precio', 'Cantidad', 'Descuentos']
        categorical_features = ['Producto', 'Categoría', 'dia_semana']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        return preprocessor