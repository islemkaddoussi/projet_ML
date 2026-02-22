Titre

Analyse du Comportement des Clients - Projet Machine Learning

Description du projet

Ce projet, Analyse du Comportement des Clients - Projet Machine Learning Retail, vise à analyser le comportement des clients d’une entreprise e-commerce spécialisée dans les cadeaux en utilisant des techniques de Machine Learning.

L’entreprise souhaite :

Comprendre le comportement des clients

Améliorer les stratégies marketing

Réduire le taux de départ des clients (churn)

Augmenter le chiffre d’affaires

Le dataset contient 52 variables (features) issues de transactions réelles de clients. Il est intentionnellement imparfait, avec des valeurs manquantes, du bruit et des données non nettoyées. Cela permet de travailler dans un contexte réaliste d’analyse de données, de prétraitement et de Machine Learning.

Structure du projet

Le projet est organisé comme suit :

projet_ml_retail/
│-- data/
│   │-- raw/
│   │-- processed/
│   └-- train_test/
│-- notebooks/
│-- src/
│   │-- preprocessing.py
│   │-- train_model.py
│   │-- predict.py
│   └-- utils.py
│-- models/
│-- app/
│-- reports/
│-- requirements.txt
│-- README.md
└-- .gitignore
Description des dossiers et fichiers

data/

data/raw/ : contient les fichiers de données brutes (par exemple CSV) avec les 52 variables décrivant les clients et leurs transactions.

data/processed/ : contient les données nettoyées et transformées après le prétraitement.

data/train_test/ : contient les données séparées en ensembles d’entraînement et de test pour la modélisation.

notebooks/
Contient les notebooks Jupyter utilisés pour :

L’analyse exploratoire des données (EDA)

Les tests de prétraitement

Les expérimentations avec les modèles et les visualisations

src/
Contient le code Python du projet :

preprocessing.py : scripts pour nettoyer les données, gérer les valeurs manquantes, encoder les variables catégorielles et normaliser les données.

train_model.py : scripts pour entraîner les modèles de Machine Learning (clustering, classification, régression).

predict.py : scripts pour générer des prédictions avec les modèles entraînés.

utils.py : fonctions utilitaires (chargement des données, sauvegarde des modèles, visualisations, etc.).

models/
Contient les modèles entraînés (fichiers .pkl ou .joblib).

app/
Contient le code pour le déploiement, notamment une future application Flask permettant d’utiliser les modèles via une interface web ou API.

reports/
Contient les rapports, graphiques et visualisations (EDA, performances des modèles, etc.).

requirements.txt
Liste toutes les dépendances Python nécessaires au projet.

README.md
Fichier de documentation principal du projet (ce fichier).

.gitignore
Liste les fichiers ignorés par Git (environnement virtuel, fichiers temporaires, etc.).

Installation

Cloner le dépôt (ou copier le projet localement) :

git clone <url-du-repository>.git
cd projet_ml_retail

Créer un environnement virtuel :

python -m venv venv

Activer l’environnement virtuel (Windows) :

venv\Scripts\activate

(Sur Linux/Mac :)

source venv/bin/activate

Installer les dépendances :

pip install --upgrade pip
pip install -r requirements.txt
Dépendances

Les dépendances du projet sont gérées avec requirements.txt.

Pour générer ou mettre à jour ce fichier :

pip freeze > requirements.txt

Cela permet de reproduire exactement le même environnement sur une autre machine.

Dataset

Le dataset est stocké dans :

data/raw/

Il contient des informations sur les clients d’une entreprise e-commerce, notamment :

Données démographiques (âge, genre, région)

Données transactionnelles (montant d’achat, fréquence, récence)

Données comportementales (type de client, fidélité, interactions)

Le dataset contient 52 variables et est intentionnellement imparfait, avec :

Valeurs manquantes

Données aberrantes (outliers)

Données non nettoyées

Cela permet de travailler sur :

L’analyse exploratoire des données

Le nettoyage des données

Le Machine Learning

Étapes du projet

Le projet est structuré en plusieurs étapes.

1 - Analyse exploratoire des données (EDA)

Objectifs :

Comprendre la structure du dataset

Vérifier les valeurs manquantes

Vérifier les types de données

Calculer des statistiques descriptives

Créer des visualisations comme :

Histogrammes

Boxplots

Matrices de corrélation

Distributions des variables

Cette étape permet d’identifier les problèmes dans les données et les premières tendances.

2 - Prétraitement des données

Objectifs :

Gestion des valeurs manquantes

Remplacement ou suppression des valeurs manquantes

Encodage des variables catégorielles

Transformation des variables texte en valeurs numériques

Normalisation des données

Mise à l’échelle des variables numériques

3 - Réduction de dimension

Objectifs :

Appliquer l’ACP (PCA)

Réduire le nombre de variables

Visualiser les clusters

Améliorer les performances des modèles

4 - Modélisation

Objectifs :

Clustering

Regrouper les clients en segments

Exemples d’algorithmes :

K-Means

Clustering hiérarchique

Classification

Prédire des catégories (ex : churn)

Régression

Prédire des valeurs numériques (ex : revenu futur)

5 - Évaluation

Objectifs :

Évaluer les performances des modèles

Exemples de métriques :

Classification :

Accuracy

Precision

Recall

Régression :

RMSE

MAE

R²

Clustering :

Silhouette Score

6 - Déploiement

Objectifs :

Créer une application Flask permettant :

D’envoyer des données clients

D’obtenir des prédictions

De visualiser les résultats

Utilisation

Après installation :

Prétraitement
python src/preprocessing.py
Entraînement
python src/train_model.py
Prédiction
python src/predict.py


Projet Machine Learning - Islem Kaddoussi GI2-S3