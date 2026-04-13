import logging
import pandas as pd
from src.data.utils import load_arff_with_unique_names

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_amazon_reviews(path: str) -> pd.DataFrame:

    logger.info(f"Cargando datos desde: {path}")

    # Cargar archivo .arff
    data, meta = load_arff_with_unique_names(path)
    df = pd.DataFrame(data)

    registros_iniciales = df.shape[0]
    logger.info(f"Registros cargados: {registros_iniciales}")

    # Convertir columna target correcta
    df['class_1'] = df['class_1'].apply(
        lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
    )

    # Eliminar columna 'class' incorrecta y renombrar
    df = df.drop(columns=['class'])
    df = df.rename(columns={'class_1': 'class'})

    # Eliminar duplicados
    df = df.drop_duplicates()
    registros_finales = df.shape[0]

    logger.info(f"Registros tras limpieza: {registros_finales}")
    logger.info(f"Registros eliminados: {registros_iniciales - registros_finales}")

    return df