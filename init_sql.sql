-- Initialisation de la base de données YOLO Detection
-- Ce script est exécuté automatiquement au premier démarrage de PostgreSQL

-- Extension pour UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table pour stocker les détections globales
CREATE TABLE IF NOT EXISTS detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    confidence_threshold FLOAT NOT NULL DEFAULT 0.5,
    processing_time_ms INTEGER NOT NULL,
    total_objects_detected INTEGER NOT NULL DEFAULT 0,
    image_width INTEGER NOT NULL,
    image_height INTEGER NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    upload_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processing_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    mlflow_run_id VARCHAR(255),
    status VARCHAR(20) DEFAULT 'completed' CHECK (status IN ('processing', 'completed', 'failed')),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table pour stocker les objets détectés individuellement
CREATE TABLE IF NOT EXISTS detected_objects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detection_id UUID NOT NULL REFERENCES detections(id) ON DELETE CASCADE,
    class_name VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_x1 FLOAT NOT NULL,
    bbox_y1 FLOAT NOT NULL,
    bbox_x2 FLOAT NOT NULL,
    bbox_y2 FLOAT NOT NULL,
    bbox_width FLOAT NOT NULL,
    bbox_height FLOAT NOT NULL,
    bbox_area FLOAT NOT NULL,
    relative_x1 FLOAT NOT NULL, -- Coordonnées relatives (0-1)
    relative_y1 FLOAT NOT NULL,
    relative_x2 FLOAT NOT NULL,
    relative_y2 FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table pour les métriques des modèles
CREATE TABLE IF NOT EXISTS model_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(50) NOT NULL,
    evaluation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    dataset_name VARCHAR(100),
    total_images INTEGER NOT NULL DEFAULT 0,
    total_detections INTEGER NOT NULL DEFAULT 0,
    avg_processing_time_ms FLOAT,
    avg_confidence FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    map_50 FLOAT, -- mAP@0.5
    map_95 FLOAT, -- mAP@0.5:0.95
    mlflow_experiment_id VARCHAR(255),
    mlflow_run_id VARCHAR(255),
    hyperparameters JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table pour les sessions utilisateur (optionnel)
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_ip INET,
    user_agent TEXT,
    total_uploads INTEGER DEFAULT 0,
    total_detections INTEGER DEFAULT 0,
    first_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Index pour optimiser les performances
CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(processing_timestamp);
CREATE INDEX IF NOT EXISTS idx_detections_model ON detections(model_name);
CREATE INDEX IF NOT EXISTS idx_detections_status ON detections(status);
CREATE INDEX IF NOT EXISTS idx_detected_objects_detection_id ON detected_objects(detection_id);
CREATE INDEX IF NOT EXISTS idx_detected_objects_class ON detected_objects(class_name);
CREATE INDEX IF NOT EXISTS idx_detected_objects_confidence ON detected_objects(confidence);
CREATE INDEX IF NOT EXISTS idx_model_metrics_model ON model_metrics(model_name);
CREATE INDEX IF NOT EXISTS idx_model_metrics_date ON model_metrics(evaluation_date);
CREATE INDEX IF NOT EXISTS idx_user_sessions_session_id ON user_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_activity ON user_sessions(last_activity);

-- Fonction pour mettre à jour automatiquement updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger pour mettre à jour automatiquement updated_at dans detections
CREATE TRIGGER update_detections_updated_at 
    BEFORE UPDATE ON detections 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Vues utiles pour les statistiques
CREATE OR REPLACE VIEW detection_summary AS
SELECT 
    DATE(processing_timestamp) as detection_date,
    model_name,
    COUNT(*) as total_detections,
    AVG(processing_time_ms) as avg_processing_time,
    AVG(total_objects_detected) as avg_objects_per_image,
    SUM(total_objects_detected) as total_objects_detected
FROM detections 
WHERE status = 'completed'
GROUP BY DATE(processing_timestamp), model_name
ORDER BY detection_date DESC, model_name;

CREATE OR REPLACE VIEW popular_classes AS
SELECT 
    class_name,
    COUNT(*) as detection_count,
    AVG(confidence) as avg_confidence,
    MIN(confidence) as min_confidence,
    MAX(confidence) as max_confidence
FROM detected_objects 
GROUP BY class_name 
ORDER BY detection_count DESC;

-- Insertion de données d'exemple (optionnel)
-- Vous pouvez décommenter ces lignes pour avoir des données de test

/*
INSERT INTO detections (
    filename, original_filename, model_name, confidence_threshold,
    processing_time_ms, total_objects_detected, image_width, image_height, file_size_bytes
) VALUES 
    ('sample1.jpg', 'test_image1.jpg', 'yolov8n', 0.5, 150, 3, 640, 480, 102400),
    ('sample2.jpg', 'test_image2.jpg', 'yolov8s', 0.6, 200, 2, 800, 600, 153600);
*/

-- Permissions pour l'utilisateur yolo_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO yolo_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO yolo_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO yolo_user;

-- Message de confirmation
DO $$
BEGIN
    RAISE NOTICE 'Base de données YOLO Detection initialisée avec succès!';
    RAISE NOTICE 'Tables créées: detections, detected_objects, model_metrics, user_sessions';
    RAISE NOTICE 'Vues créées: detection_summary, popular_classes';
    RAISE NOTICE 'Index et triggers configurés pour optimiser les performances';
END $$;