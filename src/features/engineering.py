"""
DOCUMENTACIÓN DE FEATURES
Variables de entrada:
- 9,999 columnas de frecuencias de palabras (extraídas del dataset ARFF de Amazon).
- Estas variables representan la presencia de términos específicos en las reseñas.

Procesamiento realizado:
- StandardScaler: Se aplica escalado estándar para normalizar las frecuencias.
- Filtrado: Se excluye la columna 'class' (target) del proceso de transformación.
- Consistencia: Uso de ColumnTransformer para asegurar que train e inference sean idénticos.
"""

import pandas as pd
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, target_col='class'):
        self.pipeline = None
        self.target_col = target_col

    def build_pipeline(self, df: pd.DataFrame):
        """
        Crea un pipeline que escala todas las frecuencias de palabras.
        """
        # Todas las columnas (excepto la clase) son nuestras features
        self.feature_cols = [col for col in df.columns if col != self.target_col]
        
        # En este dataset, todas las columnas son numéricas (frecuencias)
        # Se usa StandardScaler para que las palabras muy comunes no dominen el modelo
        self.pipeline = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), self.feature_cols)
            ],
            remainder='drop'
        )
        logger.info(f"Pipeline de features construido para {len(self.feature_cols)} variables.")
        return self.pipeline

    def apply_transform(self, df: pd.DataFrame, fit: bool = False):
        """Asegura que train/inference usen la misma transformación."""
        if fit:
            return self.pipeline.fit_transform(df)
        return self.pipeline.transform(df)