import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
import mlflow
import mlflow.pytorch
from PIL import Image, ImageDraw, ImageFont

class YOLODetector:
    def __init__(self, mlflow_uri=None):
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        
        self.models = {}
        self.available_models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
        
    def load_model(self, model_name='yolov8n'):
        """Charge un modèle YOLO"""
        if model_name not in self.models:
            try:
                # Essayer de charger depuis MLflow d'abord
                model_path = self._get_model_from_mlflow(model_name)
                if model_path and os.path.exists(model_path):
                    self.models[model_name] = YOLO(model_path)
                else:
                    # Charger le modèle par défaut
                    self.models[model_name] = YOLO(f'{model_name}.pt')
                    
                print(f"Modèle {model_name} chargé avec succès")
                
            except Exception as e:
                print(f"Erreur lors du chargement du modèle {model_name}: {e}")
                # Fallback vers yolov8n
                if model_name != 'yolov8n':
                    self.models[model_name] = YOLO('yolov8n.pt')
                else:
                    raise e
                    
        return self.models[model_name]
    
    def _get_model_from_mlflow(self, model_name):
        """Récupère un modèle depuis MLflow"""
        try:
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(model_name, stages=["Production"])
            
            if model_version:
                model_uri = f"models:/{model_name}/Production"
                local_path = mlflow.pytorch.load_model(model_uri)
                return local_path
                
        except Exception as e:
            print(f"Impossible de charger le modèle depuis MLflow: {e}")
            
        return None
    
    def detect(self, image, model_name='yolov8n', confidence=0.5):
        """Effectue la détection d'objets sur une image"""
        start_time = time.time()
        
        # Charger le modèle
        model = self.load_model(model_name)
        
        # Effectuer la prédiction
        results = model(image, conf=confidence)
        
        processing_time = time.time() - start_time
        
        # Traitement des résultats
        detections = []
        annotated_image = image.copy()
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i, box in enumerate(boxes):
                # Coordonnées de la boîte
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence_score = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                
                detection = {
                    'class': class_name,
                    'confidence': float(confidence_score),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                }
                detections.append(detection)
                
                # Annotation de l'image
                annotated_image = self._draw_detection(
                    annotated_image, x1, y1, x2, y2, 
                    class_name, confidence_score
                )
        
        result_data = {
            'model_name': model_name,
            'confidence_threshold': confidence,
            'processing_time': processing_time,
            'objects_count': len(detections),
            'detections': detections,
            'image_shape': image.shape if hasattr(image, 'shape') else None
        }
        
        return annotated_image, result_data
    
    def _draw_detection(self, image, x1, y1, x2, y2, class_name, confidence):
        """Dessine les détections sur l'image"""
        if isinstance(image, np.ndarray):
            # Image OpenCV
            color = (0, 255, 0)  # Vert
            thickness = 2
            
            # Dessiner le rectangle
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # Ajouter le texte
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)[0]
            
            cv2.rectangle(image, (int(x1), int(y1) - label_size[1] - 10), 
                         (int(x1) + label_size[0], int(y1)), color, -1)
            
            cv2.putText(image, label, (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness)
            
        else:
            # Image PIL
            draw = ImageDraw.Draw(image)
            
            # Dessiner le rectangle
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            
            # Ajouter le texte
            label = f"{class_name}: {confidence:.2f}"
            draw.text((x1, y1 - 15), label, fill="green")
        
        return image
    
    def train_model(self, data_path, model_name='yolov8n', epochs=100, imgsz=640):
        """Entraîne un modèle YOLO avec tracking MLflow"""
        
        experiment_name = f"yolo_training_{model_name}"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run() as run:
            # Log des paramètres
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("imgsz", imgsz)
            mlflow.log_param("data_path", data_path)
            
            # Charger le modèle
            model = YOLO(f'{model_name}.pt')
            
            # Entraînement
            start_time = time.time()
            results = model.train(
                data=data_path,
                epochs=epochs,
                imgsz=imgsz,
                save=True,
                plots=True
            )
            
            training_time = time.time() - start_time
            
            # Log des métriques
            metrics = results.results_dict
            mlflow.log_metric("training_time", training_time)
            
            if 'metrics/mAP50(B)' in metrics:
                mlflow.log_metric("map_50", metrics['metrics/mAP50(B)'])
            if 'metrics/mAP50-95(B)' in metrics:
                mlflow.log_metric("map_50_95", metrics['metrics/mAP50-95(B)'])
            
            # Sauvegarder le modèle
            model_path = f"./models/{model_name}_trained.pt"
            model.save(model_path)
            
            mlflow.log_artifact(model_path)
            mlflow.pytorch.log_model(
                pytorch_model=model.model,
                artifact_path="model",
                registered_model_name=model_name
            )
            
            return run.info.run_id, metrics
    
    def evaluate_model(self, model_name, test_data_path):
        """Évalue un modèle sur des données de test"""
        model = self.load_model(model_name)
        
        # Validation
        results = model.val(data=test_data_path)
        
        metrics = {
            'map_50': results.box.map50,
            'map_50_95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr
        }
        
        return metrics
    
    def get_available_models(self):
        """Retourne la liste des modèles disponibles"""
        return self.available_models