# Imagen base oficial de Python 3.9 (versión slim = ligera)
FROM python:3.9-slim

# Metadata de la imagen
LABEL maintainer="tu-email@ejemplo.com"
LABEL version="1.0"
LABEL description="Heart Disease Prediction API"

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivo de requirements primero (para aprovechar cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY api_heart.py .
COPY heart_model_wrapper.py .
COPY heart_disease_model.joblib .
COPY models ./models

ENV MODEL_PATH=/app/heart_disease_model.joblib
ENV MODEL_REGISTRY_PATH=/app/models/model_registry.json

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Cambiar a usuario no-root
USER appuser

# Exponer el puerto 8000
EXPOSE 8000

# Health check: verificar que la API responde
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Comando por defecto: iniciar uvicorn
CMD ["uvicorn", "api_heart:app", "--host", "0.0.0.0", "--port", "8000"]
