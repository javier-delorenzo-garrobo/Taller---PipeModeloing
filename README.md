# Heart Disease ML Deployment

Stack local para servir una API de inferencia, recoger metricas con Prometheus,
visualizarlas en Grafana y lanzar predicciones desde un frontend web.

## Arranque

```bash
docker compose up --build
```

Servicios:

- Frontend: http://localhost:8080
- API: http://localhost:8000
- Swagger: http://localhost:8000/docs o http://localhost:8080/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

Grafana se provisiona automaticamente con usuario `admin` y password `admin`.
El dashboard queda en la carpeta `ML Monitoring` con el nombre
`Heart Disease Model Metrics`.
Tambien queda configurado como dashboard de inicio al abrir
http://localhost:3000.

## Flujo de metricas

1. El frontend llama a `POST /predict`.
2. La API registra contadores, latencia, errores, confianza y distribucion de
   resultados en `/metrics`.
3. Prometheus raspa `api:8000/metrics`.
4. Grafana lee Prometheus y actualiza el dashboard cada 5 segundos.

## Versionado del modelo

Registrar y promover un nuevo modelo:

```bash
make version-model MODEL=heart_disease_model.joblib VERSION=1.0.1
```

Tambien se puede ejecutar directamente:

```bash
python3 scripts/version_model.py --source heart_disease_model.joblib --version 1.0.1 --promote
```

El script copia el artefacto a `models/`, calcula su `sha256`, actualiza
`models/model_registry.json` y deja la version promovida como modelo activo para
la API.

Para sincronizarlo con Git, el proyecto debe estar inicializado como repositorio:

```bash
git init
git add .
git commit -m "Initial ML deployment"
```

Despues puedes registrar el modelo, commitearlo y crear un tag Git con:

```bash
make version-model-git MODEL=heart_disease_model.joblib VERSION=1.0.1
```

Ese comando hace:

1. Copia el modelo a `models/heart_disease_model_v<version>_<hash>.joblib`.
2. Actualiza `models/model_registry.json`.
3. Promueve esa version copiandola a `heart_disease_model.joblib`.
4. Ejecuta `git add` sobre el artefacto, el registro y el modelo activo.
5. Crea un commit `Version model <version>`.
6. Crea un tag anotado `model-v<version>`.

Si tienes remoto configurado, sincroniza con:

```bash
git push
git push --tags
```

## Parada

```bash
docker compose down
```
