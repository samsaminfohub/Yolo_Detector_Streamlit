import streamlit as st
import os
import time
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow

from yolo_detector import YOLODetector
from database import DatabaseManager

# Configuration de la page
st.set_page_config(
    page_title="YOLO Object Detection Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation des composants
@st.cache_resource
def init_components():
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    detector = YOLODetector(mlflow_uri=mlflow_uri)
    db_manager = DatabaseManager()
    return detector, db_manager

detector, db_manager = init_components()

# Interface principale
def main():
    st.markdown('<h1 class="main-header">🔍 YOLO Object Detection Platform</h1>', unsafe_allow_html=True)
    
    # Barre latérale pour la navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choisir une page",
        ["🏠 Accueil", "📸 Détection", "📊 Analyse", "🤖 Modèles", "📈 MLflow"]
    )
    
    if page == "🏠 Accueil":
        show_home_page()
    elif page == "📸 Détection":
        show_detection_page()
    elif page == "📊 Analyse":
        show_analysis_page()
    elif page == "🤖 Modèles":
        show_models_page()
    elif page == "📈 MLflow":
        show_mlflow_page()

def show_home_page():
    """Page d'accueil avec statistiques générales"""
    st.header("📊 Tableau de bord")
    
    # Récupération des statistiques
    stats = db_manager.get_detection_stats()
    
    if stats:
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Détections", stats.get('total_detections', 0))
        
        with col2:
            st.metric("Total Objets", stats.get('total_objects', 0))
        
        with col3:
            avg_objects = stats.get('total_objects', 0) / max(stats.get('total_detections', 1), 1)
            st.metric("Moyenne Objets/Image", f"{avg_objects:.1f}")
        
        with col4:
            st.metric("Modèles Utilisés", len(stats.get('model_usage', [])))
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            if stats.get('top_classes'):
                df_classes = pd.DataFrame(stats['top_classes'])
                fig = px.bar(df_classes, x='class', y='count', 
                           title="Classes les plus détectées",
                           color='count', color_continuous_scale='viridis')
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if stats.get('model_usage'):
                df_models = pd.DataFrame(stats['model_usage'])
                fig = px.pie(df_models, values='count', names='model',
                           title="Utilisation des modèles")
                st.plotly_chart(fig, use_container_width=True)
    
    # Historique récent
    st.header("📋 Historique récent")
    history = db_manager.get_detection_history(limit=10)
    
    if not history.empty:
        st.dataframe(
            history[['filename', 'upload_time', 'model_name', 
                    'objects_detected', 'processing_time']],
            use_container_width=True
        )
    else:
        st.info("Aucune détection enregistrée pour le moment.")

def show_detection_page():
    """Page de détection d'objets"""
    st.header("📸 Détection d'objets")
    
    # Paramètres de détection
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Paramètres")
        
        # Sélection du modèle
        model_name = st.selectbox(
            "Modèle YOLO",
            detector.get_available_models(),
            index=0
        )
        
        # Seuil de confiance
        confidence = st.slider(
            "Seuil de confiance",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        # Sauvegarde en base
        save_to_db = st.checkbox("Sauvegarder en base", value=True)
    
    with col1:
        st.subheader("Upload d'image")
        
        # Upload de fichier
        uploaded_file = st.file_uploader(
            "Choisir une image",
            type=['png', 'jpg', 'jpeg'],
            help="Formats supportés: PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            # Affichage de l'image originale
            image = Image.open(uploaded_file)
            st.image(image, caption="Image originale", use_column_width=True)
            
            # Bouton de détection
            if st.button("🔍 Détecter les objets", type="primary"):
                with st.spinner("Détection en cours..."):
                    # Conversion en format numpy pour YOLO
                    image_np = np.array(image)
                    
                    # Détection
                    annotated_image, results = detector.detect(
                        image_np, model_name=model_name, confidence=confidence
                    )
                    
                    # Affichage des résultats
                    st.success(f"✅ Détection terminée en {results['processing_time']:.2f}s")
                    
                    # Image annotée
                    st.subheader("Résultats de détection")
                    st.image(annotated_image, caption="Image avec détections", use_column_width=True)
                    
                    # Statistiques
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Objets détectés", results['objects_count'])
                    with col2:
                        st.metric("Temps de traitement", f"{results['processing_time']:.2f}s")
                    with col3:
                        st.metric("Modèle utilisé", model_name)
                    
                    # Détails des objets détectés
                    if results['detections']:
                        st.subheader("Objets détectés")
                        detections_df = pd.DataFrame(results['detections'])
                        detections_df['confidence'] = detections_df['confidence'].round(3)
                        st.dataframe(detections_df[['class', 'confidence']], use_container_width=True)
                    
                    # Sauvegarde en base de données
                    if save_to_db:
                        image_width, image_height = image.size
                        detection_id = db_manager.save_detection_result(
                            filename=uploaded_file.name,
                            model_name=model_name,
                            confidence_threshold=confidence,
                            objects_detected=results['objects_count'],
                            processing_time=results['processing_time'],
                            image_width=image_width,
                            image_height=image_height,
                            results_data=results
                        )
                        
                        if detection_id:
                            st.success(f"✅ Résultats sauvegardés (ID: {detection_id})")
                        else:
                            st.error("❌ Erreur lors de la sauvegarde")

def show_analysis_page():
    """Page d'analyse des données"""
    st.header("📊 Analyse des données")
    
    # Récupération des données
    history = db_manager.get_detection_history(limit=100)
    
    if history.empty:
        st.info("Aucune donnée disponible pour l'analyse.")
        return
    
    # Métriques de performance
    st.subheader("⚡ Métriques de performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_time = history['processing_time'].mean()
        st.metric("Temps moyen", f"{avg_time:.2f}s")
    
    with col2:
        avg_objects = history['objects_detected'].mean()
        st.metric("Objets/image", f"{avg_objects:.1f}")
    
    with col3:
        total_images = len(history)
        st.metric("Images traitées", total_images)
    
    with col4:
        unique_models = history['model_name'].nunique()
        st.metric("Modèles utilisés", unique_models)
    
    # Graphiques d'analyse
    col1, col2 = st.columns(2)
    
    with col1:
        # Évolution temporelle
        st.subheader("📈 Évolution temporelle")
        history['date'] = pd.to_datetime(history['upload_time']).dt.date
        daily_stats = history.groupby('date').agg({
            'objects_detected': 'sum',
            'processing_time': 'mean'
        }).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=daily_stats['date'], y=daily_stats['objects_detected'],
                      name="Objets détectés", line=dict(color='blue')),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=daily_stats['date'], y=daily_stats['processing_time'],
                      name="Temps moyen (s)", line=dict(color='red')),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Objets détectés", secondary_y=False)
        fig.update_yaxes(title_text="Temps de traitement (s)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribution des temps de traitement
        st.subheader("⏱️ Distribution des temps")
        fig = px.histogram(history, x='processing_time', nbins=20,
                          title="Distribution des temps de traitement")
        fig.update_xaxes(title_text="Temps de traitement (s)")
        fig.update_yaxes(title_text="Fréquence")
        st.plotly_chart(fig, use_container_width=True)
    
    # Analyse par modèle
    st.subheader("🤖 Analyse par modèle")
    
    model_stats = history.groupby('model_name').agg({
        'processing_time': ['mean', 'std'],
        'objects_detected': ['mean', 'sum'],
        'filename': 'count'
    }).round(3)
    
    model_stats.columns = ['Temps moyen', 'Écart-type temps', 'Objets/image', 'Total objets', 'Nb images']
    st.dataframe(model_stats, use_container_width=True)
    
    # Graphique comparatif des modèles
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(history, x='model_name', y='processing_time',
                    title="Temps de traitement par modèle")
        fig.update_xaxes(title_text="Modèle")
        fig.update_yaxes(title_text="Temps (s)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(history, x='model_name', y='objects_detected',
                    title="Objets détectés par modèle")
        fig.update_xaxes(title_text="Modèle")
        fig.update_yaxes(title_text="Nombre d'objets")
        st.plotly_chart(fig, use_container_width=True)

def show_models_page():
    """Page de gestion des modèles"""
    st.header("🤖 Gestion des modèles")
    
    # Métriques des modèles
    metrics_df = db_manager.get_model_metrics()
    
    if not metrics_df.empty:
        st.subheader("📊 Métriques des modèles")
        
        # Sélection du modèle à afficher
        selected_model = st.selectbox(
            "Sélectionner un modèle",
            metrics_df['model_name'].unique()
        )
        
        model_data = metrics_df[metrics_df['model_name'] == selected_model].iloc[-1]
        
        # Affichage des métriques
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Précision", f"{model_data['precision_avg']:.3f}")
        with col2:
            st.metric("Rappel", f"{model_data['recall_avg']:.3f}")
        with col3:
            st.metric("F1-Score", f"{model_data['f1_score_avg']:.3f}")
        with col4:
            st.metric("mAP@0.5", f"{model_data['map_50']:.3f}")
        
        # Comparaison des modèles
        st.subheader("🏆 Comparaison des modèles")
        
        comparison_metrics = ['precision_avg', 'recall_avg', 'f1_score_avg', 'map_50', 'map_50_95']
        
        fig = go.Figure()
        
        for metric in comparison_metrics:
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=metrics_df['model_name'],
                y=metrics_df[metric]
            ))
        
        fig.update_layout(
            title="Comparaison des métriques par modèle",
            xaxis_title="Modèle",
            yaxis_title="Score",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau détaillé
        st.subheader("📋 Détails des modèles")
        display_df = metrics_df[['model_name', 'experiment_name', 'accuracy', 
                               'precision_avg', 'recall_avg', 'f1_score_avg', 
                               'map_50', 'created_at']].copy()
        display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(display_df, use_container_width=True)
    
    else:
        st.info("Aucune métrique de modèle disponible.")
    
    # Section d'entraînement (simulée)
    st.subheader("🔧 Entraînement de modèles")
    
    with st.expander("Paramètres d'entraînement"):
        col1, col2 = st.columns(2)
        
        with col1:
            train_model = st.selectbox(
                "Modèle de base",
                detector.get_available_models()
            )
            epochs = st.number_input("Nombre d'époques", min_value=1, max_value=300, value=100)
        
        with col2:
            img_size = st.selectbox("Taille d'image", [416, 640, 1024], index=1)
            experiment_name = st.text_input("Nom de l'expérience", "custom_training")
        
        if st.button("🚀 Lancer l'entraînement"):
            st.info("Fonctionnalité d'entraînement en cours de développement.")
            st.info("En production, cette section permettrait de lancer l'entraînement avec vos propres données.")

def show_mlflow_page():
    """Page MLflow"""
    st.header("📈 MLflow Tracking")
    
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    
    st.info(f"Interface MLflow disponible à l'adresse: {mlflow_uri}")
    
    # Bouton pour ouvrir MLflow
    if st.button("🔗 Ouvrir MLflow UI"):
        st.markdown(f"[Ouvrir MLflow]({mlflow_uri})")
    
    # Informations sur MLflow
    st.subheader("ℹ️ À propos de MLflow")
    
    st.markdown("""
    MLflow est une plateforme open-source pour la gestion du cycle de vie du machine learning.
    
    **Fonctionnalités disponibles:**
    - 📊 Tracking des expériences et métriques
    - 📦 Gestion des modèles et versions
    - 🔄 Déploiement de modèles
    - 📈 Comparaison des runs
    
    **Dans cette application:**
    - Les métriques d'entraînement sont automatiquement loggées
    - Les modèles sont versionnés et stockés
    - Interface web pour visualiser les expériences
    """)
    
    # Essayer de récupérer des informations depuis MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        
        experiments = mlflow.search_experiments()
        
        if experiments:
            st.subheader("🧪 Expériences MLflow")
            
            exp_data = []
            for exp in experiments:
                exp_data.append({
                    'Nom': exp.name,
                    'ID': exp.experiment_id,
                    'Statut': exp.lifecycle_stage,
                    'Créé': exp.creation_time
                })
            
            if exp_data:
                exp_df = pd.DataFrame(exp_data)
                st.dataframe(exp_df, use_container_width=True)
    
    except Exception as e:
        st.warning(f"Impossible de se connecter à MLflow: {e}")
        st.info("Assurez-vous que le service MLflow est démarré.")

if __name__ == "__main__":
    main()