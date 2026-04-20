# 🛍️ Retail Churn Predictor — Analyse Comportementale Clientèle

> Atelier Machine Learning · Module GI2 · Année universitaire 2025-2026  
> Préparé par Fadoua Drira

Chaîne complète de traitement ML appliquée à un dataset e-commerce de cadeaux :  
**Exploration → Préparation → Modélisation → Évaluation → Déploiement**

---

## 📁 Structure du projet

```
projet_ml_retail/
├── data/
│   ├── raw/                    # Données brutes originales (.csv)
│   ├── processed/              # Données nettoyées (cleaned_data.csv)
│   └── train_test/             # Splits train/test + échantillon brut
├── notebooks/                  # Notebooks Jupyter (prototypage / EDA)
├── src/
│   ├── preprocessing.py        # Pipeline de prétraitement complet
│   ├── train_model.py          # Entraînement des modèles (4 classifieurs)
│   ├── predict.py              # Prédiction sur nouvelles données
│   └── utils.py                # Fonctions utilitaires (plots, save…)
├── models/                     # Artefacts sauvegardés
│   ├── imputer.joblib
│   ├── scaler.joblib
│   ├── pca.joblib
│   ├── churn_classifier.pkl    # Meilleur modèle de classification
│   ├── monetary_regressor.pkl  # RandomForest régresseur
│   └── customer_clusters.pkl  # KMeans (4 segments)
├── app/
│   └── app.py                  # Application web Flask
├── reports/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── model_comparison.csv
│   └── predictions.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/<votre-username>/projet_ml_retail.git
cd projet_ml_retail
```

### 2. Créer et activer l'environnement virtuel

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 🚀 Guide d'utilisation

### Étape 1 — Prétraitement

```bash
python src/preprocessing.py
```

Ce script :
- Charge `data/raw/retail_customers_COMPLETE_CATEGORICAL.csv`
- Effectue le feature engineering (IP, dates, ratios)
- Encode les variables catégorielles (One-Hot + Ordinal + Target Encoding)
- Supprime les outliers via IsolationForest (contamination=6%)
- Applique KNNImputer → StandardScaler → PCA (10 composantes)
- Sauvegarde les artefacts dans `models/` et les splits dans `data/train_test/`

### Étape 2 — Entraînement

```bash
python src/train_model.py
```

Ce script :
- Compare 4 modèles : Decision Tree, KNN, Random Forest, XGBoost
- Utilise SMOTE pour rééquilibrer les classes
- Optimise les hyperparamètres via GridSearchCV (cv=5, scoring=F1)
- Sauvegarde le meilleur classifieur, un régresseur MonetaryTotal et un KMeans
- Génère la matrice de confusion et la courbe ROC dans `reports/`

### Étape 3 — Prédiction batch

```bash
python src/predict.py
```

Applique le pipeline complet sur `data/train_test/X_test_brut_40.csv` et sauvegarde les résultats dans `reports/predictions.csv`.

### Étape 4 — Interface web Flask

```bash
python app/app.py
```

Ouvrir [http://127.0.0.1:5000](http://127.0.0.1:5000) dans votre navigateur.

| Route | Description |
|-------|-------------|
| `GET /` | Formulaire de saisie manuelle d'un client |
| `POST /predict` | Prédiction (HTML ou JSON selon Content-Type) |
| `GET /batch` | Prédiction sur les 40 clients de test |
| `GET /health` | Vérification que les modèles sont chargés |

**Appel API (JSON) :**

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Recency": 45, "Frequency": 3, "MonetaryAvg": 60, "Age": 28}'
```

---

## 🧠 Modèles & Résultats

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| Decision Tree | — | — | — | — | — |
| KNN | — | — | — | — | — |
| **Random Forest** | — | — | — | — | — |
| XGBoost | — | — | — | — | — |

> Les valeurs seront complétées après exécution de `train_model.py` et reportées dans `reports/model_comparison.csv`.

### Pipeline de prétraitement

```
Données brutes (52 features)
    ↓  Feature Engineering (IP, dates, ratios)
    ↓  One-Hot + Ordinal + Target Encoding
    ↓  IsolationForest (suppression outliers 6%)
    ↓  Train / Test split (80/20, stratifié)
    ↓  KNNImputer (k=5)
    ↓  StandardScaler
    ↓  PCA (10 composantes)
    ↓  SMOTE (rééquilibrage)
    ↓  GridSearchCV (5 folds)
```

---

## 📦 Dépendances principales

| Package | Usage |
|---------|-------|
| `pandas`, `numpy` | Manipulation des données |
| `scikit-learn` | Preprocessing, modèles, évaluation |
| `xgboost` | Classifieur gradient boosting |
| `imbalanced-learn` | SMOTE |
| `flask` | Interface web |
| `joblib` | Sérialisation des modèles |
| `matplotlib`, `seaborn` | Visualisations |

---

## 👤 Auteur

**Atelier ML — GI2**  
Encadrant : Fadoua Drira  
Année : 2025-2026