# 🔍 YOLO Object Detection Platform

Plateforme complète de détection d'objets utilisant YOLO avec interface Streamlit, base de données PostgreSQL et tracking MLflow.

## 🏗️ Architecture

```
├── app.py                    # Application Streamlit principale
├── yolo_detector.py         # Classe de détection YOLO
├── database.py              # Gestionnaire de base de données
├── requirements.txt         # Dépendances Python
├── docker-compose.yml       # Configuration Docker Compose
├── Dockerfile.streamlit     # Dockerfile pour Streamlit
├── Dockerfile.mlflow        # Dockerfile pour MLflow
├── init.sql                 # Script d'initialisation PostgreSQL
└── README.md               # Ce fichier
```

## 🚀 Démarrage rapide

### Prérequis
- Docker et Docker Compose installés
- Au moins 4GB de RAM disponible
- Connexion Internet pour télécharger les modèles YOLO

### Installation

1. **Cloner le projet** (ou créer les fichiers manuellement)
```bash
git clone <votre-repo>
cd yolo-detection-platform
```

2. **Créer la structure des dossiers**
```bash
mkdir -p uploads models
```

3. **Lancer les services**
```bash
docker-compose up -d
```

4. **Vérifier le démarrage**
```bash
docker-compose ps
```

### Accès aux services

- **🖥️ Interface Streamlit**: http://localhost:8501
- **📊 MLflow UI**: http://localhost:5000
- **🗃️ PostgreSQL**: localhost:5432

## 📋 Fonctionnalités

### 🏠 Tableau de bord
- Statistiques générales des détections
- Graphiques des classes les plus détectées
- Historique des traitements récents
- Métriques de performance

### 📸 Détection d'objets
- Upload d'images (PNG, JPG, JPEG)
- Choix du modèle YOLO (n, s, m, l, x)
- Ajustement du seuil de confiance
- Visualisation des résultats annotés
- Sauvegarde automatique en base de données

### 📊 Analyse des données
- Métriques de performance par modèle
- Évolution temporelle des détections
- Distribution des temps de traitement
- Comparaison des modèles

### 🤖 Gestion des modèles
- Métriques détaillées (précision, rappel, F1-score, mAP)
- Comparaison visuelle des performances
- Interface d'entraînement (à développer)

### 📈 MLflow Integration
- Tracking automatique des expériences
- Versioning des modèles
- Interface web pour la gestion

## 🛠️ Configuration avancée

### Variables d'environnement

Vous pouvez modifier les variables dans le `docker-compose.yml`:

```yaml
environment:
  - DATABASE_URL=postgresql://yolo_user:yolo_password@postgres:5432/yolo_db
  - MLFLOW_TRACKING_URI=http://mlflow:5000
```

### Modèles YOLO disponibles

- **YOLOv8n**: Nano - Plus rapide, moins précis
- **YOLOv8s**: Small - Bon équilibre
- **YOLOv8m**: Medium - Plus précis
- **YOLOv8l**: Large - Très précis
- **YOLOv8x**: Extra Large - Maximum de précision

### Base de données

Le schéma inclut:
- `detections`: Résultats de détection avec métadonnées
- `detected_objects`: Objets individuels détectés
- `model_metrics`: Métriques des modèles entraînés

## 🔧 Développement

### Exécution en local

1. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

2. **Démarrer PostgreSQL et MLflow**
```bash
docker-compose up postgres mlflow -d
```

3. **Lancer Streamlit**
```bash
streamlit run app.py
```

### Ajout de nouveaux modèles

Pour intégrer un nouveau modèle:

1. Modifier `yolo_detector.py` pour ajouter le modèle
2. Mettre à jour la liste dans `get_available_models()`
3. Tester la compatibilité avec MLflow

### Personnalisation de l'interface

L'interface Streamlit peut être personnalisée via:
- CSS personnalisé dans `app.py`
- Modification des layouts et composants
- Ajout de nouvelles pages de visualisation

## 📊 Métriques et monitoring

### Base de données
- Temps de traitement par image
- Nombre d'objets détectés
- Utilisation des modèles
- Historique complet

### MLflow
- Métriques d'entraînement
- Paramètres des expériences
- Artifacts des modèles
- Comparaison des runs

## 🚨 Dépannage

### Problèmes courants

**Erreur de connexion à la base de données**
```bash
# Vérifier que PostgreSQL est démarré
docker-compose logs postgres

# Redémarrer si nécessaire
docker-compose restart postgres
```

**Modèle YOLO non trouvé**
```bash
# Les modèles se téléchargent automatiquement
# Vérifier l'espace disque et la connexion Internet
df -h
```

**Interface Streamlit inaccessible**
```bash
# Vérifier les logs
docker-compose logs streamlit

# Redémarrer le service
docker-compose restart streamlit
```

### Logs et debugging

```bash
# Voir tous les logs
docker-compose logs

# Logs d'un service spécifique
docker-compose logs streamlit
docker-compose logs mlflow
docker-compose logs postgres

# Suivre les logs en temps réel
docker-compose logs -f streamlit
```

## 🔒 Sécurité

### Recommandations pour la production

1. **Changer les mots de passe par défaut**
2. **Utiliser HTTPS avec un reverse proxy**
3. **Configurer des sauvegardes automatiques**
4. **Limiter l'accès réseau aux services**
5. **Surveiller l'utilisation des ressources**

### Backup de la base de données

```bash
# Créer un backup
docker-compose exec postgres pg_dump -U yolo_user yolo_db > backup.sql

# Restaurer un backup
docker-compose exec -T postgres psql -U yolo_user yolo_db < backup.sql
```

## 📈 Performance

### Optimisations recommandées

1. **GPU Support**: Ajouter le support CUDA pour les modèles YOLO
2. **Cache Redis**: Pour mettre en cache les résultats fréquents
3. **Load Balancer**: Pour distribuer la charge sur plusieurs instances
4. **CDN**: Pour servir les images statiques

### Monitoring des ressources

```bash
# Utilisation des conteneurs
docker stats

# Espace disque
docker system df
```

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à:
- Ouvrir des issues pour signaler des bugs
- Proposer de nouvelles fonctionnalités
- Améliorer la documentation
- Optimiser les performances

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## 🆘 Support

Pour obtenir de l'aide:
1. Consulter cette documentation
2. Vérifier les logs avec `docker-compose logs`
3. Ouvrir une issue sur le repository GitHub

---

**Développé avec ❤️ en utilisant YOLO, Streamlit, PostgreSQL et MLflow**