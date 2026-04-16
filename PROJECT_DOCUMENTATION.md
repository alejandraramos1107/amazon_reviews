# Amazon Reviews Authorship Classification

## 1. Resumen del proyecto

Este proyecto implementa un pipeline de machine learning para clasificar la autoría de reseñas de Amazon a partir de un dataset vectorizado con 9,999 features. El objetivo fue construir una solución reproducible y orquestada que permitiera:

- cargar y validar los datos
- transformar las features
- entrenar varios modelos de clasificación
- optimizar hiperparámetros
- registrar experimentos en MLflow
- orquestar el flujo con Prefect
- seleccionar un modelo candidato
- desplegar una interfaz sencilla de inferencia
- empaquetar la solución con Docker y desplegarla en AWS

## 2. Objetivo del problema

El problema es de clasificación multiclase. Cada registro del dataset representa una reseña ya transformada en un vector de frecuencias de palabras, y la variable objetivo (`class`) corresponde al autor de la reseña.

El resultado esperado del modelo es:

- entrada: vector de 9,999 features
- salida: autor predicho

## 3. Arquitectura general

El proyecto quedó dividido en cinco capas:

### Configuración

Archivos:

- `src/config/constants.py`
- `src/config/mlflow_setup.py`
- `src/config/__init__.py`

Responsabilidades:

- rutas del proyecto
- configuración de MLflow
- constantes del dataset
- parámetros de entrenamiento
- parámetros de Optuna
- nombres de flow y deployment de Prefect

### Datos

Archivos:

- `src/data/loaders.py`
- `src/data/validators.py`
- `src/data/utils.py`

Responsabilidades:

- cargar el archivo `.arff`
- limpiar la columna target
- eliminar duplicados
- validar esquema, tamaño, nulos y consistencia

### Features

Archivo:

- `src/features/engineering.py`

Responsabilidades:

- construir el pipeline de transformación
- escalar las features numéricas
- asegurar consistencia entre entrenamiento e inferencia

### Modelos

Archivos:

- `src/models/trainer.py`
- `src/models/evaluate.py`
- `src/models/model_registry.py`
- `src/models/__init__.py`

Responsabilidades:

- optimización con Optuna
- entrenamiento de modelos
- evaluación
- selección del mejor candidato
- exportación del mejor modelo para serving

### Orquestación y serving

Archivos:

- `pipeline.py`
- `deploy.py`
- `app.py`
- `src/serving/model_loader.py`
- `templates/index.html`
- `Dockerfile`

Responsabilidades:

- orquestar el flujo con Prefect
- publicar el deployment
- servir una interfaz web de predicción
- empaquetar la app en Docker

## 4. Flujo del pipeline

El flujo principal quedó definido en `pipeline.py` y sigue este orden:

1. Configuración de MLflow
2. Carga del dataset
3. Validación del dataset
4. Separación train/test
5. Feature engineering
6. Optimización de hiperparámetros con Optuna
7. Entrenamiento de modelos finales
8. Comparación de métricas
9. Selección del mejor modelo
10. Exportación del mejor modelo a `serving_artifacts/`
11. Creación de artefacto final en Prefect

## 5. Modelos evaluados

Se compararon tres modelos:

- `MultinomialNB`
- `LogisticRegression`
- `LinearSVC`

### Justificación

- `MultinomialNB` es un baseline natural para datos tipo bag-of-words
- `LogisticRegression` suele funcionar bien en clasificación de texto
- `LinearSVC` es un modelo lineal fuerte para problemas de alta dimensión

## 6. Métrica principal

La métrica principal elegida fue `f1_macro`.

### ¿Por qué `f1_macro`?

Porque el problema tiene aproximadamente 50 clases y se quería evaluar el desempeño de manera equilibrada entre todos los autores. A diferencia de `accuracy`, `f1_macro` calcula el F1 de cada clase por separado y luego obtiene un promedio, evitando que unas pocas clases dominen la evaluación.

También se registró `accuracy` como métrica complementaria.

## 7. Optimización con Optuna

Inicialmente el tuning era demasiado costoso porque se estaba aplicando a varios modelos y en un espacio de búsqueda muy amplio. Después de analizar los tiempos de ejecución, se dejó una configuración más realista:

- `OPTUNA_TRIALS = 2`
- `OPTUNA_CV_FOLDS = 2`
- tuning solo para `LogisticRegression`

### Ajustes para reducir tiempo

Se hicieron estos cambios:

- se redujo el número de trials
- se redujo el número de folds
- se dejó `MultinomialNB` como baseline fijo
- se dejó `LinearSVC` como baseline fijo
- se quitó `solver='saga'` del espacio de búsqueda de `LogisticRegression`

### Motivo

El cuello de botella principal estaba en:

- `LinearSVC`
- algunos trials costosos de `LogisticRegression`

La demora se debía a:

- 9,999 features
- 50 clases
- validación cruzada
- optimización iterativa de modelos lineales

## 8. Resultados del entrenamiento

En una de las ejecuciones finales del pipeline, los resultados fueron:

- `MultinomialNB`
  - `f1_macro = 0.7616`
  - `accuracy = 0.7568`
- `LogisticRegression`
  - `f1_macro = 0.8465`
  - `accuracy = 0.8480`
- `LinearSVC`
  - `f1_macro = 0.8226`
  - `accuracy = 0.8243`
  - tiempo de entrenamiento muy alto

## 9. Modelo candidato

El modelo candidato seleccionado fue `LogisticRegression`.

### Justificación

- obtuvo el mejor `f1_macro`
- obtuvo la mejor `accuracy`
- tuvo un costo computacional mucho menor que `LinearSVC`
- ofreció el mejor balance entre desempeño y tiempo

## 10. Tracking en MLflow

Se utilizó MLflow para:

- registrar parámetros
- registrar métricas
- guardar artefactos del modelo
- comparar experimentos

El experimento utilizado fue:

- `amazon-reviews-authorship`

MLflow permitió visualizar:

- trials de Optuna
- runs de entrenamiento final
- métricas por modelo
- artefactos del mejor candidato

## 11. Orquestación con Prefect

Se utilizó Prefect para:

- orquestar el pipeline completo
- ejecutar el flow por etapas
- visualizar logs
- publicar un deployment reutilizable

El flow principal quedó con el nombre:

- `Amazon Reviews Classification Pipeline`

El deployment quedó con el nombre:

- `amazon-reviews-training`

### Función del deployment

El deployment no hace inferencia directa. Su función es publicar el pipeline completo en Prefect para que pueda ejecutarse desde la UI o programarse con schedules.

Por eso, cuando se corre el deployment, se vuelve a ejecutar todo el pipeline de entrenamiento:

- carga
- validación
- features
- Optuna
- entrenamiento
- selección del mejor modelo

## 12. Interfaz web de inferencia

Se construyó una interfaz sencilla con FastAPI y Jinja2.

Archivos:

- `app.py`
- `templates/index.html`
- `src/serving/model_loader.py`

### ¿Qué hace?

- carga el mejor modelo exportado localmente
- permite seleccionar una fila del dataset por índice
- realiza la predicción del autor
- muestra:
  - índice
  - autor predicho
  - autor real
  - si acertó
  - métricas del mejor modelo

### Nota importante

La interfaz no recibe texto libre porque el dataset ya está vectorizado. Para permitir texto libre habría que reconstruir el vocabulario original y el proceso de vectorización.

## 13. Exportación del mejor modelo

Para que el serving con Docker fuera limpio, se decidió no depender de MLflow en tiempo de ejecución dentro del contenedor.

En su lugar:

- el pipeline exporta el mejor modelo a `serving_artifacts/`
- también se guarda un `metadata.json`

Archivos clave:

- `src/models/model_registry.py`
- `export_best_model.py`

Esto separa correctamente:

- entrenamiento y tracking
- serving e inferencia

## 14. Dockerización

Se creó un `Dockerfile` para empaquetar la app web.

Archivos:

- `Dockerfile`
- `.dockerignore`

### Flujo de uso

1. Exportar el mejor modelo
2. Construir la imagen Docker
3. Levantar el contenedor
4. Exponer la app en el puerto `8000`

## 15. Despliegue en AWS

La aplicación de inferencia se desplegó en una instancia EC2.

Pasos realizados:

1. Crear instancia EC2
2. Instalar Docker
3. Instalar Git
4. Clonar el repositorio
5. Construir la imagen Docker
6. Ejecutar el contenedor
7. Abrir el puerto en el Security Group

### Resultado

La interfaz web quedó accesible desde la IP pública de la instancia.

## 16. Diferencia entre entrenamiento y serving

Es importante distinguir dos componentes:

### Pipeline de entrenamiento

- corre con Prefect
- usa MLflow
- compara modelos
- selecciona el mejor candidato

### App de inferencia

- corre con FastAPI
- carga el mejor modelo ya exportado
- hace predicciones
- es la parte desplegada con Docker y AWS

## 17. Problemas encontrados y soluciones

### Compatibilidades de entorno

Se resolvieron incompatibilidades entre:

- `mlflow`
- `protobuf`
- `setuptools`
- `pandas`

### Tiempos excesivos de entrenamiento

Se identificó que:

- `LinearSVC` era el principal cuello de botella
- algunos trials de `LogisticRegression` también eran costosos

Se resolvió:

- reduciendo tuning
- simplificando el espacio de búsqueda
- dejando un flujo más realista para desarrollo local

### Serving con Docker y MLflow

Inicialmente la app intentaba cargar el modelo directamente desde MLflow dentro del contenedor, pero esto generaba problemas con las rutas de artefactos.

La solución fue:

- exportar el mejor modelo a un directorio local
- hacer que la app de inferencia cargue el modelo desde `serving_artifacts/`

## 18. Conclusión

El proyecto logró construir un flujo completo de MLOps que incluye:

- preparación y validación de datos
- feature engineering
- entrenamiento y comparación de modelos
- optimización de hiperparámetros
- tracking con MLflow
- orquestación con Prefect
- selección del mejor candidato
- interfaz web de inferencia
- Dockerización
- despliegue en AWS

El modelo candidato final fue `LogisticRegression`, seleccionado por su mejor balance entre desempeño y costo computacional.

## 19. Archivos principales del proyecto

- `pipeline.py`: flujo principal de entrenamiento
- `deploy.py`: publicación del deployment en Prefect
- `app.py`: interfaz web de inferencia
- `export_best_model.py`: exportación del mejor modelo
- `src/config/*`: configuración
- `src/data/*`: carga y validación
- `src/features/engineering.py`: transformación de features
- `src/models/*`: entrenamiento, evaluación y exportación
- `src/serving/model_loader.py`: carga del modelo para la app
- `templates/index.html`: interfaz HTML
- `Dockerfile`: empaquetado para despliegue
