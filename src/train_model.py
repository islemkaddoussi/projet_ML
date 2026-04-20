import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             mean_squared_error, r2_score)
import joblib
from utils import save_model

os.makedirs('reports', exist_ok=True)

def plot_cm(y_true, y_pred, title, filename):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matrice de Confusion : {title}')
    plt.ylabel('Réel')
    plt.xlabel('Prédit')
    plt.tight_layout()
    plt.savefig(f'reports/{filename}')
    plt.close()

def plot_roc(y_true, y_proba, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title(f'Courbe ROC : {title}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'reports/{filename}')
    plt.close()

def plot_metrics_bar(metrics_dict, title, filename):
    """Graphique en barres des métriques pour un modèle"""
    plt.figure(figsize=(8, 5))
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    bars = plt.bar(names, values, color=colors[:len(names)], edgecolor='black')
    plt.ylim(0, 1.05)
    plt.title(f'Métriques : {title}')
    plt.ylabel('Score')
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'reports/{filename}')
    plt.close()

def main():
    print("=== ENTRAÎNEMENT ET VISUALISATION ===")

    # ============================================================
    # 1. CHARGEMENT
    # ============================================================
    X_train = pd.read_csv('data/train_test/X_train_pca.csv')
    X_test = pd.read_csv('data/train_test/X_test_pca.csv')
    y_train_class = pd.read_csv('data/train_test/y_train.csv')['Churn']
    y_test_class = pd.read_csv('data/train_test/y_test.csv')['Churn']
    
    y_train_reg = pd.read_csv('data/train_test/y_reg_train.csv')['MonetaryTotal']
    y_test_reg = pd.read_csv('data/train_test/y_reg_test.csv')['MonetaryTotal']

    X_train_scaled = pd.read_csv('data/train_test/X_train_scaled.csv')
    X_test_scaled = pd.read_csv('data/train_test/X_test_scaled.csv')

    # ============================================================
    # 2. CLASSIFICATION
    # ============================================================
    print("\n🔹 CLASSIFICATION")
    
    models = {
        'DecisionTree': (DecisionTreeClassifier(random_state=42), 
                        {'clf__max_depth': [5, 10, None]}),
        'KNN': (KNeighborsClassifier(), 
                {'clf__n_neighbors': [3, 5, 7]}),
        'RandomForest': (RandomForestClassifier(class_weight='balanced', random_state=42),
                        {'clf__n_estimators': [100, 200], 'clf__max_depth': [10, None]}),
        'XGBoost': (XGBClassifier(random_state=42, eval_metric='logloss'),
                   {'clf__n_estimators': [100, 200], 'clf__max_depth': [3, 5]})
    }

    results = {}
    roc_data = {}
    best_f1 = 0
    best_model = None
    best_name = None

    for name, (model, params) in models.items():
        print(f"   Training {name}...")
        
        pipe = Pipeline([('smote', SMOTE(random_state=42)), ('clf', model)])
        grid = GridSearchCV(pipe, params, cv=5, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train_class)
        
        y_pred = grid.predict(X_test)
        y_proba = grid.predict_proba(X_test)[:, 1]
        
        # Calcul de TOUTES les métriques
        acc = accuracy_score(y_test_class, y_pred)
        prec = precision_score(y_test_class, y_pred, zero_division=0)
        rec = recall_score(y_test_class, y_pred, zero_division=0)
        f1 = f1_score(y_test_class, y_pred)
        auc = roc_auc_score(y_test_class, y_proba)
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        }
        
        print(f"      Accuracy={acc:.3f} | Precision={prec:.3f} | Recall={rec:.3f} | F1={f1:.3f} | AUC={auc:.3f}")
        
        # 1. Matrice de confusion
        plot_cm(y_test_class, y_pred, name, f'cm_{name}.png')
        
        # 2. Courbe ROC
        fpr, tpr, _ = roc_curve(y_test_class, y_proba)
        roc_data[name] = (fpr, tpr, auc)
        plot_roc(y_test_class, y_proba, name, f'roc_{name}.png')
        
        # 3. ✅ NOUVEAU : Barres des métriques par modèle
        plot_metrics_bar(results[name], name, f'metrics_{name}.png')
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = grid.best_estimator_
            best_name = name

    # ============================================================
    # COMPARAISON GLOBALE
    # ============================================================
    print(f"\n   🏆 Meilleur modèle : {best_name} (F1={best_f1:.3f})")
    save_model(best_model, 'models/churn_classifier.pkl')

    # Tableau comparatif
    df_results = pd.DataFrame(results).T
    print("\n📊 Tableau comparatif :")
    print(df_results.round(3))

    # Graphique comparatif global
    df_results.plot(kind='bar', figsize=(12, 6), colormap='viridis', edgecolor='black')
    plt.title('Comparaison des Métriques par Modèle')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=0)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=5)
    plt.tight_layout()
    plt.savefig('reports/metrics_comparison.png')
    plt.close()

    # Courbes ROC superposées
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Comparaison des Courbes ROC')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('reports/roc_all_models.png')
    plt.close()

    # ============================================================
    # 3. RÉGRESSION
    # ============================================================
    print("\n🔹 RÉGRESSION")
    reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg.fit(X_train_scaled, y_train_reg)
    
    y_pred_reg = reg.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    r2 = r2_score(y_test_reg, y_pred_reg)
    print(f"   RMSE={rmse:.2f} | R²={r2:.3f}")
    
    save_model(reg, 'models/monetary_regressor.pkl')
    
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test_reg, y_pred_reg, alpha=0.5, color='teal', edgecolors='black', linewidth=0.5)
    min_val = min(y_test_reg.min(), y_pred_reg.min())
    max_val = max(y_test_reg.max(), y_pred_reg.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Prédiction parfaite')
    plt.xlabel('Valeur Réelle (MonetaryTotal)')
    plt.ylabel('Valeur Prédite')
    plt.title(f'Régression : Réel vs Prédit (R²={r2:.3f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/regression_scatter.png')
    plt.close()

    # ============================================================
    # 4. CLUSTERING
    # ============================================================
    print("\n🔹 CLUSTERING")
    inertias = []
    K_range = range(2, 11)
    
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_train)
        inertias.append(km.inertia_)
    
    # Méthode du coude
    p1, p2 = np.array([K_range[0], inertias[0]]), np.array([K_range[-1], inertias[-1]])
    line_vec = p2 - p1
    distances = []
    for i, k in enumerate(K_range):
        p3 = np.array([k, inertias[i]])
        cross = abs((p3[0]-p1[0])*line_vec[1] - (p3[1]-p1[1])*line_vec[0])
        distances.append(cross / np.linalg.norm(line_vec))
    
    optimal_k = K_range[np.argmax(distances)]
    print(f"   Coude détecté à K={optimal_k}")

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertias, 'bo-', markerfacecolor='r', markersize=8)
    plt.axvline(x=optimal_k, color='green', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Inertie (SSE)')
    plt.title('Méthode du Coude')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/elbow_curve.png')
    plt.close()

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(X_train)
    save_model(kmeans, 'models/customer_clusters.pkl')
    
    # Visualisation 2D (PC1 vs PC2)
    clusters = kmeans.predict(X_train)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], 
                         c=clusters, cmap='viridis', alpha=0.6, edgecolors='w', s=50)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'Segmentation Clients (K={optimal_k})')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig('reports/clusters_2d.png')
    plt.close()

    print("\n✅ Tous les entraînements et visualisations terminés")
    print("📁 Graphiques générés dans le dossier 'reports/'")

if __name__ == "__main__":
    main()