-- Création des tables pour l'application YOLO

-- Table pour stocker les résultats de détection
CREATE TABLE IF NOT EXISTS detections (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_name VARCHAR(100) NOT NULL,
    confidence_threshold FLOAT DEFAULT 0.5,
    objects_detected INTEGER DEFAULT 0,
    processing_time FLOAT,
    image_width INTEGER,
    image_height INTEGER,
    results_json JSONB
);

-- Table pour stocker les objets détectés individuellement
CREATE TABLE IF NOT EXISTS detected_objects (
    id SERIAL PRIMARY KEY,
    detection_id INTEGER REFERENCES detections(id) ON DELETE CASCADE,
    class_name VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_x1 FLOAT NOT NULL,
    bbox_y1 FLOAT NOT NULL,
    bbox_x2 FLOAT NOT NULL,
    bbox_y2 FLOAT NOT NULL,
    bbox_width FLOAT,
    bbox_height FLOAT
);

-- Table pour les métriques des modèles
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    run_id VARCHAR(255),
    experiment_name VARCHAR(255),
    accuracy FLOAT,
    precision_avg FLOAT,
    recall_avg FLOAT,
    f1_score_avg FLOAT,
    map_50 FLOAT,
    map_50_95 FLOAT,
    training_time FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour optimiser les requêtes
CREATE INDEX IF NOT EXISTS idx_detections_filename ON detections(filename);
CREATE INDEX IF NOT EXISTS idx_detections_model ON detections(model_name);
CREATE INDEX IF NOT EXISTS idx_detections_upload_time ON detections(upload_time);
CREATE INDEX IF NOT EXISTS idx_detected_objects_class ON detected_objects(class_name);
CREATE INDEX IF NOT EXISTS idx_model_metrics_name ON model_metrics(model_name);

-- Insertion de données de test
INSERT INTO model_metrics (model_name, experiment_name, accuracy, precision_avg, recall_avg, f1_score_avg, map_50, map_50_95) 
VALUES 
    ('yolov8n', 'baseline_experiment', 0.89, 0.85, 0.82, 0.83, 0.78, 0.65),
    ('yolov8s', 'improved_experiment', 0.92, 0.88, 0.86, 0.87, 0.82, 0.70),
    ('yolov8m', 'production_experiment', 0.94, 0.91, 0.89, 0.90, 0.85, 0.73);