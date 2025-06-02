import os
import json
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd

class DatabaseManager:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL', 'postgresql://yolo_user:yolo_password@localhost:5432/yolo_db')
        self.engine = create_engine(self.database_url)
        self.Session = sessionmaker(bind=self.engine)
    
    def save_detection_result(self, filename, model_name, confidence_threshold, 
                            objects_detected, processing_time, image_width, 
                            image_height, results_data):
        """Sauvegarde les résultats de détection dans la base de données"""
        try:
            with self.engine.connect() as conn:
                # Insertion dans la table detections
                query = text("""
                    INSERT INTO detections (filename, model_name, confidence_threshold, 
                                          objects_detected, processing_time, image_width, 
                                          image_height, results_json)
                    VALUES (:filename, :model_name, :confidence_threshold, 
                            :objects_detected, :processing_time, :image_width, 
                            :image_height, :results_json)
                    RETURNING id
                """)
                
                result = conn.execute(query, {
                    'filename': filename,
                    'model_name': model_name,
                    'confidence_threshold': confidence_threshold,
                    'objects_detected': objects_detected,
                    'processing_time': processing_time,
                    'image_width': image_width,
                    'image_height': image_height,
                    'results_json': json.dumps(results_data)
                })
                
                detection_id = result.fetchone()[0]
                
                # Insertion des objets détectés individuellement
                if 'detections' in results_data:
                    for detection in results_data['detections']:
                        obj_query = text("""
                            INSERT INTO detected_objects (detection_id, class_name, confidence,
                                                        bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                                                        bbox_width, bbox_height)
                            VALUES (:detection_id, :class_name, :confidence,
                                    :bbox_x1, :bbox_y1, :bbox_x2, :bbox_y2,
                                    :bbox_width, :bbox_height)
                        """)
                        
                        conn.execute(obj_query, {
                            'detection_id': detection_id,
                            'class_name': detection['class'],
                            'confidence': detection['confidence'],
                            'bbox_x1': detection['bbox'][0],
                            'bbox_y1': detection['bbox'][1],
                            'bbox_x2': detection['bbox'][2],
                            'bbox_y2': detection['bbox'][3],
                            'bbox_width': detection['bbox'][2] - detection['bbox'][0],
                            'bbox_height': detection['bbox'][3] - detection['bbox'][1]
                        })
                
                conn.commit()
                return detection_id
                
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
            return None
    
    def get_detection_history(self, limit=50):
        """Récupère l'historique des détections"""
        try:
            query = text("""
                SELECT id, filename, upload_time, model_name, confidence_threshold,
                       objects_detected, processing_time, image_width, image_height
                FROM detections 
                ORDER BY upload_time DESC 
                LIMIT :limit
            """)
            
            return pd.read_sql(query, self.engine, params={'limit': limit})
            
        except Exception as e:
            print(f"Erreur lors de la récupération: {e}")
            return pd.DataFrame()
    
    def get_detection_stats(self):
        """Récupère les statistiques globales"""
        try:
            with self.engine.connect() as conn:
                stats = {}
                
                # Nombre total de détections
                result = conn.execute(text("SELECT COUNT(*) FROM detections"))
                stats['total_detections'] = result.fetchone()[0]
                
                # Nombre total d'objets détectés
                result = conn.execute(text("SELECT COUNT(*) FROM detected_objects"))
                stats['total_objects'] = result.fetchone()[0]
                
                # Classes les plus détectées
                result = conn.execute(text("""
                    SELECT class_name, COUNT(*) as count 
                    FROM detected_objects 
                    GROUP BY class_name 
                    ORDER BY count DESC 
                    LIMIT 10
                """))
                stats['top_classes'] = [{'class': row[0], 'count': row[1]} for row in result.fetchall()]
                
                # Modèles les plus utilisés
                result = conn.execute(text("""
                    SELECT model_name, COUNT(*) as count 
                    FROM detections 
                    GROUP BY model_name 
                    ORDER BY count DESC
                """))
                stats['model_usage'] = [{'model': row[0], 'count': row[1]} for row in result.fetchall()]
                
                return stats
                
        except Exception as e:
            print(f"Erreur lors du calcul des stats: {e}")
            return {}
    
    def get_model_metrics(self):
        """Récupère les métriques des modèles"""
        try:
            query = text("""
                SELECT model_name, experiment_name, accuracy, precision_avg,
                       recall_avg, f1_score_avg, map_50, map_50_95, created_at
                FROM model_metrics 
                ORDER BY created_at DESC
            """)
            
            return pd.read_sql(query, self.engine)
            
        except Exception as e:
            print(f"Erreur lors de la récupération des métriques: {e}")
            return pd.DataFrame()
    
    def save_model_metrics(self, model_name, run_id, experiment_name, metrics):
        """Sauvegarde les métriques d'un modèle"""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    INSERT INTO model_metrics (model_name, run_id, experiment_name,
                                             accuracy, precision_avg, recall_avg, f1_score_avg,
                                             map_50, map_50_95, training_time)
                    VALUES (:model_name, :run_id, :experiment_name,
                            :accuracy, :precision_avg, :recall_avg, :f1_score_avg,
                            :map_50, :map_50_95, :training_time)
                """)
                
                conn.execute(query, {
                    'model_name': model_name,
                    'run_id': run_id,
                    'experiment_name': experiment_name,
                    'accuracy': metrics.get('accuracy'),
                    'precision_avg': metrics.get('precision_avg'),
                    'recall_avg': metrics.get('recall_avg'),
                    'f1_score_avg': metrics.get('f1_score_avg'),
                    'map_50': metrics.get('map_50'),
                    'map_50_95': metrics.get('map_50_95'),
                    'training_time': metrics.get('training_time')
                })
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des métriques: {e}")
            return False