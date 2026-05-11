"""
Heart Disease Model Wrapper
============================

Este módulo contiene el wrapper del modelo de predicción de enfermedad cardíaca.
Encapsula toda la lógica de preprocesamiento, predicción y postprocesamiento.

Autor: Ejercicio Práctico - Despliegue ML
Fecha: 2026
"""

import pandas as pd
import numpy as np


class HeartDiseaseWrapper:
    """
    Wrapper para el modelo de predicción de enfermedad cardíaca.
    
    Encapsula toda la lógica de inferencia:
    - Preprocesamiento de entrada
    - Predicción del modelo
    - Postprocesamiento de salida
    
    Attributes:
        model: Pipeline de scikit-learn entrenado
        feature_names: Lista con los nombres exactos de las características
    """
    
    def __init__(self, model):
        """
        Inicializa el wrapper con un modelo entrenado.
        
        Args:
            model: Pipeline de scikit-learn entrenado
        """
        self.model = model
        # Nombres exactos de las características del dataset original
        self.feature_names = [
            'age', 'sex', 'chest', 'resting_blood_pressure',
            'serum_cholestoral', 'fasting_blood_sugar', 
            'resting_electrocardiographic_results',
            'maximum_heart_rate_achieved', 'exercise_induced_angina', 
            'oldpeak', 'slope', 'number_of_major_vessels', 'thal'
        ]
    
    def preprocess(self, input_data):
        """
        Preprocesa los datos de entrada.
        
        Args:
            input_data: dict o DataFrame con las características
            
        Returns:
            DataFrame listo para el modelo
            
        Raises:
            ValueError: Si faltan características requeridas
        """
        if isinstance(input_data, dict):
            # Convertir dict a DataFrame
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Validar que todas las características estén presentes
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Faltan características: {missing_features}")
        
        # Ordenar columnas según el orden esperado
        df = df[self.feature_names]
        
        # Convertir a numérico
        df = df.apply(pd.to_numeric, errors='coerce')
        
        return df
    
    def predict(self, input_data):
        """
        Realiza predicción sobre los datos de entrada.
        
        Args:
            input_data: dict o DataFrame con las características
            
        Returns:
            dict con predicción y probabilidades en formato:
            {
                'prediction': int,
                'prediction_label': str,
                'confidence': float,
                'probabilities': dict,
                'risk_level': str
            }
        """
        # Preprocesar
        df = self.preprocess(input_data)
        
        # Predecir
        prediction = self.model.predict(df)[0]
        probabilities = self.model.predict_proba(df)[0]
        
        # Postprocesar
        result = self.postprocess(prediction, probabilities)
        
        return result
    
    def predict_batch(self, input_data_list):
        """
        Realiza predicciones batch sobre múltiples entradas.
        
        Args:
            input_data_list: lista de dicts o DataFrame
            
        Returns:
            lista de dicts con predicciones
        """
        if isinstance(input_data_list, list):
            df = pd.DataFrame(input_data_list)
        else:
            df = input_data_list
        
        # Preprocesar
        df = self.preprocess(df)
        
        # Predecir
        predictions = self.model.predict(df)
        probabilities = self.model.predict_proba(df)
        
        # Postprocesar cada resultado
        results = []
        for pred, probs in zip(predictions, probabilities):
            results.append(self.postprocess(pred, probs))
        
        return results
    
    def postprocess(self, prediction, probabilities):
        """
        Postprocesa la salida del modelo en un formato amigable.
        
        Args:
            prediction: clase predicha (0 o 1)
            probabilities: probabilidades de cada clase
            
        Returns:
            dict con resultado interpretado
        """
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Disease' if prediction == 1 else 'No Disease',
            'confidence': float(probabilities[prediction]),
            'probabilities': {
                'no_disease': float(probabilities[0]),
                'disease': float(probabilities[1])
            },
            'risk_level': self._get_risk_level(probabilities[1])
        }
        return result
    
    def _get_risk_level(self, disease_probability):
        """
        Determina el nivel de riesgo basado en la probabilidad.
        
        Args:
            disease_probability: probabilidad de enfermedad
            
        Returns:
            str: nivel de riesgo ('Low', 'Medium', 'High')
        """
        if disease_probability < 0.3:
            return 'Low'
        elif disease_probability < 0.7:
            return 'Medium'
        else:
            return 'High'
    
    def get_feature_info(self):
        """
        Retorna información sobre las características esperadas.
        
        Returns:
            dict con información de las características
        """
        return {
            'feature_names': self.feature_names,
            'num_features': len(self.feature_names),
            'description': 'Heart Disease prediction features from OpenML dataset'
        }