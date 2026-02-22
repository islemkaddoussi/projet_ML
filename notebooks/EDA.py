import pandas as pd           # Manipulation de données
import matplotlib.pyplot as plt  # Graphiques
import seaborn as sns        # Graphiques statistiques
import numpy as np           # Calculs numériques
import os

# =============================================================================
# CONFIGURATION CHEMIN ABSOLU 

BASE_DIR = r"C:\Users\Islam\Documents\projet_ML"
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Créer le dossier reports s'il n'existe pas
os.makedirs(REPORTS_DIR, exist_ok=True)
print(f"📁 Dossier reports: {REPORTS_DIR}")

# =============================================================================
# CHARGEMENT DES DONNÉES

df = pd.read_csv("C:\\Users\\Islam\\Documents\\projet_ML\\data\\raw\\retail_customers_COMPLETE_CATEGORICAL.csv")
print(f"Taille du dataset : {df.shape}")

# Vérification simple des dates (sans parsing complet)
if 'RegistrationDate' in df.columns:
    print(f"\nExemple RegistrationDate: {df['RegistrationDate'].head(3).tolist()}")
    print("Format: texte (parsing au preprocessing)")

# =============================================================================
# INFORMATIONS GENERALES

print("\n=== INFO DATASET ===")
print(df.info())

print("\n=== STATISTIQUES ===")
print(df.describe())

print("\nColonnes :", df.columns.tolist())

# =============================================================================
# DOUBLONS

duplicates = df.duplicated().sum()
print(f"\nDoublons détectés : {duplicates}")

# =============================================================================
# VALEURS MANQUANTES

missing = df.isnull().sum()
missing = missing[missing > 0]

print("\n=== VALEURS MANQUANTES ===")
if len(missing) > 0:
    print(missing)
else:
    print("Pas de valeurs manquantes visibles")

# =============================================================================
# VALEURS SUSPECTES (cachées)

suspect_values = [-1, 999, 99, "Unknown", "unknown", "NA", ""]

print("\n=== VALEURS SUSPECTES ===")
for col in df.select_dtypes(include=['object', 'number']).columns:
    for val in suspect_values:
        count = (df[col] == val).sum()
        if count > 0:
            print(f"⚠️ {col}: {count} valeurs '{val}'")

# =============================================================================
# HISTOGRAMMES 

important_cols = ['Recency', 'Frequency', 'MonetaryTotal', 'Age']
important_cols = [c for c in important_cols if c in df.columns]

if len(important_cols) > 0:
    df[important_cols].hist(figsize=(12, 8), bins=30, color='skyblue', edgecolor='black')
    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, "histograms.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Sauvegardé: {save_path}")
    
    plt.show()

# =============================================================================
# CORRELATIONS 

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if 'CustomerID' in numeric_cols:
    numeric_cols.remove('CustomerID')

if len(numeric_cols) > 0:
    if len(numeric_cols) > 10:
        priority = ['Recency', 'Frequency', 'MonetaryTotal', 'MonetaryAvg',
                    'CustomerTenureDays', 'Age', 'Churn']
        corr_cols = [c for c in priority if c in numeric_cols][:10]
    else:
        corr_cols = numeric_cols

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title("Corrélations")
    plt.tight_layout()
    
    # CORRECTION: Chemin absolu
    save_path = os.path.join(REPORTS_DIR, "correlations.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Sauvegardé: {save_path}")
    
    plt.show()

# =============================================================================
# BOXPLOTS 

outlier_cols = ['MonetaryTotal', 'Age', 'Recency']
outlier_cols = [c for c in outlier_cols if c in df.columns]

if len(outlier_cols) > 0:
    fig, axes = plt.subplots(1, len(outlier_cols), figsize=(5*len(outlier_cols), 5))
    if len(outlier_cols) == 1:
        axes = [axes]

    for idx, col in enumerate(outlier_cols):
        sns.boxplot(y=df[col], ax=axes[idx], color='lightcoral')
        axes[idx].set_title(col)

    plt.tight_layout()
    
    # CORRECTION: Chemin absolu
    save_path = os.path.join(REPORTS_DIR, "boxplots.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Sauvegardé: {save_path}")
    
    plt.show()

# =============================================================================
# ANALYSE MÉTIER 

if 'Frequency' in df.columns and 'MonetaryTotal' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Frequency', y='MonetaryTotal', alpha=0.5)
    plt.title("Relation Fréquence vs Montant")
    
    # CORRECTION: Chemin absolu
    save_path = os.path.join(REPORTS_DIR, "scatter_rfm.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Sauvegardé: {save_path}")
    
    plt.show()

# =============================================================================
# ANALYSE CATÉGORIELLE 

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print("\nColonnes catégorielles:", cat_cols)

if 'RFMSegment' in df.columns:
    print("\nDistribution RFMSegment:")
    print(df['RFMSegment'].value_counts())

    plt.figure(figsize=(10, 5))
    df['RFMSegment'].value_counts().plot(kind='bar', color='teal')
    plt.title("Distribution RFMSegment")
    plt.tight_layout()
    
    # CORRECTION: Chemin absolu
    save_path = os.path.join(REPORTS_DIR, "rfm_segment.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Sauvegardé: {save_path}")
    
    plt.show()

# =============================================================================
# SYNTHÈSE

print("\n" + "="*50)
print("RÉSUMÉ FINAL")
print("="*50)

print(f"Total lignes : {len(df)}")
print(f"Total colonnes : {len(df.columns)}")
print(f"Colonnes avec manquants : {len(missing)}")
print(f"Doublons : {duplicates}")
print(f"\n📁 Graphiques sauvegardés dans: {REPORTS_DIR}")
print("   - histograms.png")
print("   - correlations.png")
print("   - boxplots.png")
print("   - scatter_rfm.png")
print("   - rfm_segment.png")