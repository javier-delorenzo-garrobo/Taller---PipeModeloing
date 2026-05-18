# Resumen general del proyecto

Este proyecto no se centra solo en entrenar un modelo, sino en montar un sistema completo alrededor de él para poder usarlo de forma parecida a como se haría en un entorno real.

La aplicación permite introducir los datos de un paciente desde una interfaz web y obtener una predicción sobre posible enfermedad cardíaca. A partir de ahí, el sistema también se encarga de exponer la API, mostrar el estado del servicio, registrar métricas y mantener controlada la versión del modelo que se está usando.

## Cómo está organizado

El sistema está dividido en varias partes:

- `api_heart.py`: contiene la API hecha con FastAPI.
- `heart_model_wrapper.py`: encapsula la lógica de inferencia del modelo.
- `frontend/`: contiene la interfaz web con HTML, CSS y JavaScript.
- `monitoring/`: incluye la configuración de Prometheus y Grafana.
- `models/`: guarda los artefactos versionados del modelo y el registro de versiones.
- `scripts/version_model.py`: automatiza el versionado y promoción de modelos.

La idea es que cada parte tenga una responsabilidad clara y no mezclar toda la lógica en un único archivo.

## Flujo general

El funcionamiento normal es:

1. El usuario rellena el formulario del frontend.
2. El frontend envía los datos a la API con una petición `POST /predict`.
3. FastAPI valida los datos recibidos.
4. La API llama al wrapper, que prepara la entrada y ejecuta la predicción.
5. La respuesta vuelve al frontend con la clase estimada, la confianza, las probabilidades, el nivel de riesgo, el tiempo de inferencia y la versión del modelo.
6. Al mismo tiempo se actualizan las métricas que luego recoge Prometheus y muestra Grafana.

## La API

La API está pensada como el centro del sistema. Además de predecir, ofrece endpoints para saber si el servicio está funcionando, consultar información del modelo y exponer métricas.

Los endpoints principales son:

- `/health`: comprueba que el modelo está cargado y que la API está lista.
- `/info`: devuelve información técnica del modelo y de sus variables de entrada.
- `/model/version`: indica qué versión del modelo está activa.
- `/predict`: hace una predicción individual.
- `/predict/batch`: permite enviar varios pacientes en una sola petición.
- `/metrics`: publica las métricas para Prometheus.

La validación de datos se hace con Pydantic. Esto ayuda a que no lleguen entradas incorrectas al modelo y hace que la API sea más robusta.

## Monitorización

Una parte importante del trabajo es que el sistema no solo responde, sino que también se puede observar.

La API genera métricas como:

- número total de predicciones,
- errores,
- latencia,
- predicciones activas,
- distribución de resultados,
- nivel de riesgo estimado,
- versión del modelo activo.

Prometheus recoge esas métricas cada pocos segundos y Grafana las muestra en un dashboard. Esto permite ver si la API está funcionando bien y cómo se está comportando el sistema con el uso.

## Despliegue

Todo el proyecto se puede levantar con Docker Compose. Hay cuatro servicios principales:

- API,
- frontend,
- Prometheus,
- Grafana.

Con esto se consigue que el proyecto sea reproducible y que no dependa tanto de la configuración concreta de una máquina. Además, la API tiene un `healthcheck`, así que el resto de servicios pueden saber cuándo está realmente preparada.

## Versionado del modelo

El modelo activo no se maneja como un archivo suelto sin control. Hay un script que:

1. calcula el hash del modelo,
2. lo copia con un nombre versionado,
3. actualiza el registro JSON,
4. y, si se indica, lo marca como versión activa.

Esto permite saber qué modelo se está utilizando en cada momento y deja una base sencilla para trabajar con varias versiones sin perder trazabilidad.

## Idea final

Lo más importante del proyecto es el paso de un modelo entrenado a un sistema utilizable:

- se puede consumir desde una interfaz,
- se puede desplegar de forma reproducible,
- se puede supervisar,
- y se puede mantener con cierto control de versiones.

En ese sentido, el modelo es solo una parte del trabajo. La mayor parte del valor está en haber construido todo lo necesario para poder usarlo como un servicio real.
