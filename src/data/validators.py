import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_ROWS = 1480
EXPECTED_FEATURES = 9999
TARGET_COLUMN = 'class'
EXPECTED_AUTHORS = 50


def validate_schema(df: pd.DataFrame) -> bool:
    """Valida que el DataFrame tenga la estructura esperada."""
    errors = []

    # Validar columna target
    if TARGET_COLUMN not in df.columns:
        errors.append(f"Falta la columna target: {TARGET_COLUMN}")

    # Validar número de features
    n_features = df.shape[1] - 1
    if n_features != EXPECTED_FEATURES:
        errors.append(f"Features esperadas: {EXPECTED_FEATURES}, encontradas: {n_features}")

    # Validar tipos de datos
    numeric_cols = df.drop(columns=[TARGET_COLUMN]).select_dtypes(exclude='number')
    if not numeric_cols.empty:
        errors.append(f"Columnas no numéricas encontradas: {numeric_cols.columns.tolist()}")

    if errors:
        for e in errors:
            logger.error(f"Error de esquema: {e}")
        return False

    logger.info("Validación de esquema: OK")
    return True


def validate_nulls(df: pd.DataFrame) -> bool:
    """Valida que no haya valores nulos."""
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        logger.error(f"Se encontraron {nulls} valores nulos")
        return False
    logger.info("Validación de nulos: OK")
    return True


def validate_ranges(df: pd.DataFrame) -> bool:
    """Valida que los valores numéricos sean no negativos."""
    X = df.drop(columns=[TARGET_COLUMN])
    if (X < 0).any().any():
        logger.error("Se encontraron valores negativos en las features")
        return False
    logger.info("Validación de rangos: OK")
    return True


def validate_authors(df: pd.DataFrame) -> bool:
    """Valida que haya exactamente 50 autores."""
    n_authors = df[TARGET_COLUMN].nunique()
    if n_authors != EXPECTED_AUTHORS:
        logger.error(f"Autores esperados: {EXPECTED_AUTHORS}, encontrados: {n_authors}")
        return False
    logger.info(f"Validación de autores: OK ({n_authors} autores)")
    return True


def run_all_validations(df: pd.DataFrame) -> bool:
    """Corre todas las validaciones y retorna True si todas pasan."""
    logger.info("Iniciando validaciones...")
    validations = [
        validate_schema(df),
        validate_nulls(df),
        validate_ranges(df),
        validate_authors(df),
    ]
    if all(validations):
        logger.info(f"Todas las validaciones pasaron. Registros listos: {df.shape[0]}")
        return True
    else:
        logger.error("Algunas validaciones fallaron")
        return False