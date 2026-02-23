import pandas as pd
import numpy as np

import os
from sklearn.preprocessing import StandardScaler

BASE_DIR = r"C:\Users\Islam\Documents\projet_ML"
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")  # DÉPLACÉ ICI

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True) 

df = pd.read_csv(r"C:\Users\Islam\Documents\projet_ML\data\raw\retail_customers_COMPLETE_CATEGORICAL.csv")

#============================
# ÉTAPE 1: PARSING DES DATES

print("\n📅 Étape 1: Parsing des dates...")
print("AVANT:", df['RegistrationDate'].head(3).tolist())
df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'], dayfirst=True, errors='coerce')
# EXTRACTION year/month/day
df['RegYear'] = df['RegistrationDate'].dt.year
df['RegMonth'] = df['RegistrationDate'].dt.month
df['RegDay'] = df['RegistrationDate'].dt.day

# SUPPRESSION de la colonne originale
df = df.drop(columns=['RegistrationDate'])

print("APRÈS:", df[['RegYear', 'RegMonth', 'RegDay']].head(3).to_string())
print("✅ Dates traitées")

#============================
#ÉTAPE 2: AGE - 30% MANQUANTS

print("\n🔧 Étape 2: Imputation Age...")

# AVANT: Vérifier
missing_before = df['Age'].isnull().sum()
print(f"Manquants AVANT: {missing_before} ({missing_before/len(df)*100:.1f}%)")

# MÉTHODE : Médiane
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age) 
# APRÈS: Vérifier
missing_after = df['Age'].isnull().sum()
print(f"Manquants APRÈS: {missing_after}")
print(f"Médiane utilisée: {median_age:.1f} ans")

print("✅ Age traité")

#============================
# ÉTAPE 3: VALEURS ABERRANTES 

print("\n🔧 Étape 3: Correction valeurs aberrantes...")

#  SupportTicketsCount → Médiane
if 'SupportTicketsCount' in df.columns:
    # Compter les aberrants
    aberrant = df['SupportTicketsCount'].isin([999, -1]).sum()
    print(f"SupportTicketsCount: {aberrant} valeurs aberrantes (-1, 999)")
    
    # Remplacer par NaN puis médiane
    df['SupportTicketsCount'] = df['SupportTicketsCount'].replace([999, -1], np.nan)
    median_val = df['SupportTicketsCount'].median()
    df['SupportTicketsCount'] = df['SupportTicketsCount'].fillna(median_val)
    print(f"  → Remplacés par médiane: {median_val}")

# SatisfactionScore → Médiane
if 'SatisfactionScore' in df.columns:
    aberrant = df['SatisfactionScore'].isin([-1, 99]).sum()
    print(f"SatisfactionScore: {aberrant} valeurs aberrantes (-1, 99)")
    
    df['SatisfactionScore'] = df['SatisfactionScore'].replace([-1, 99], np.nan)
    median_val = df['SatisfactionScore'].median()
    df['SatisfactionScore'] = df['SatisfactionScore'].fillna(median_val)
    print(f"  → Remplacés par médiane: {median_val}")

print("✅ Étape 3 terminée")

#======================================
# ÉTAPE 4: SUPPRESSION FEATURES INUTILES 

print("\n🗑️ Étape 4: Suppression features inutiles...")

# Vérifier NewsletterSubscribed
if 'NewsletterSubscribed' in df.columns:
    unique_vals = df['NewsletterSubscribed'].unique()
    n_unique = len(unique_vals)
    
    print(f"NewsletterSubscribed: {n_unique} valeur(s) unique(s)")
    print(f"  Valeurs: {unique_vals}")
    
    # Si constante (1 seule valeur) → supprimer
    if n_unique == 1:
        df = df.drop(columns=['NewsletterSubscribed'])
        print("  → SUPPRIMÉE (constante)")
    else:
        print("  → CONSERVÉE (pas constante)")
print("✅ Étape 4 terminée")

# SUPPRESSION CUSTOMERID 
print("\n" + "="*50)
print("AJOUT: SUPPRESSION CUSTOMERID")
print("="*50)

if 'CustomerID' in df.columns:
    print(f"   CustomerID présent: {df['CustomerID'].nunique()} valeurs uniques")
    df_clean = df.drop(columns=['CustomerID'])
    print(f"   → CustomerID SUPPRIMÉ")
    print(f"   Dimensions: {df.shape} → {df_clean.shape}")
else:
    print("   CustomerID déjà absent")
    df_clean = df.copy()

print("✅ Étape 4 terminée")

#==============================
#ÉTAPE 5: EXTRACTION LastLoginIP

print("\n🔧 Étape 5: Extraction LastLoginIP...")

if 'LastLoginIP' in df.columns:
    # AVANT
    print("Exemples IP:", df['LastLoginIP'].head(3).tolist())
    
    # EXTRACTION: premier octet de l'IP
    # "192.168.1.45" → "192"
    df['IP_Prefix'] = df['LastLoginIP'].str.split('.').str[0]
    df['IP_Prefix'] = pd.to_numeric(df['IP_Prefix'], errors='coerce')
    
    # SUPPRESSION colonne originale
    df = df.drop(columns=['LastLoginIP'])
    
    # APRÈS
    print(f"IP_Prefix: {df['IP_Prefix'].nunique()} valeurs uniques")
    print("Exemples:", df['IP_Prefix'].head(3).tolist())

print("✅ Étape 5 terminée")

#=================================
# ÉTAPE 6: MULTICOLINÉARITÉ (seuil)

print("\n🔗 Étape 6: Multicolinéarité...")

# Vérifier corrélation MonetaryTotal vs MonetaryAvg
if 'MonetaryTotal' in df.columns and 'MonetaryAvg' in df.columns:
    corr = df['MonetaryTotal'].corr(df['MonetaryAvg'])
    print(f"MonetaryTotal ↔ MonetaryAvg: r={corr:.3f}")
    
    if abs(corr) > 0.8:
        df = df.drop(columns=['MonetaryAvg'])
        print("  → MonetaryAvg SUPPRIMÉE")
    else:
        print("  → Les deux conservées (r < 0.8)")

print("✅ Étape 6 terminée")

#==============================================
# ÉTAPE 7: VÉRIFICATION DÉSÉQUILIBRE (detection)

print("\n⚖️ Étape 7: Vérification déséquilibre classes...")

# CHURN (0 = fidèle, 1 = parti)

print("\n--- Churn ---")
churn_counts = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100

print(f"   0 (fidèle):  {churn_counts[0]} clients ({churn_pct[0]:.1f}%)")
print(f"   1 (parti):   {churn_counts[1]} clients ({churn_pct[1]:.1f}%)")

if churn_pct.max() > 80:
    print("   ⚠️ Déséquilibre SÉVÈRE")
else:
    print("   ✅ Déséquilibre MODÉRÉ")

#  ACCOUNTSTATUS 

if 'AccountStatus' in df.columns:
    print("\n--- AccountStatus ---")
    acc_counts = df['AccountStatus'].value_counts()
    acc_pct = df['AccountStatus'].value_counts(normalize=True) * 100
    
    print(acc_counts)
    print(f"\nPourcentages:")
    for status, pct in acc_pct.items():
        print(f"   {status}: {pct:.1f}%")
    
    # Vérifier quasi-constante (>95%)
    if acc_pct.max() > 95:
        print("   ⚠️ Quasi-constante → Suppression possible")
    else:
        print("   ✅ Distribution OK")

print("\n✅ Étape 7 terminée")

#=============================
# ÉTAPE 8: FEATURE ENGINEERING 

print("\n🔨 Étape 8: Création nouvelles features...")

# MonetaryPerDay = MonetaryTotal / (Recency + 1)
# Évite division par zéro avec +1
if 'MonetaryTotal' in df.columns and 'Recency' in df.columns:
    df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    print("   MonetaryPerDay = MonetaryTotal / (Recency + 1)")

# AvgBasketValue = MonetaryTotal / Frequency
if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
    df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency']
    print("   AvgBasketValue = MonetaryTotal / Frequency")

# TenureRatio = Recency / CustomerTenureDays
if 'Recency' in df.columns and 'CustomerTenureDays' in df.columns:
    df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)
    print("   TenureRatio = Recency / (CustomerTenureDays + 1)")

print(f"\n✅ Étape 8 terminée - Nouvelles dimensions: {df.shape}")

#========================
# ÉTAPE 9: STANDARDSCALER 

print("\n📏 Étape 9: Standardisation...")


# Séparer Churn (ne pas le standardiser)
if 'Churn' in df.columns:
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    print("   Churn séparé (pas de standardisation)")
else:
    X = df
    y = None

# Identifier colonnes à standardiser (pas les binaires 0/1)
cols_to_scale = []
for col in X.select_dtypes(include=[np.number]).columns:
    unique_vals = set(X[col].dropna().unique())
    if not unique_vals.issubset({0, 1, 0., 1.}):
        cols_to_scale.append(col)

print(f"   À standardiser: {len(cols_to_scale)} colonnes")

# Standardisation
scaler = StandardScaler()
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
print("   StandardScaler appliqué")

# Réassembler
if y is not None:
    df = pd.concat([X, y], axis=1)
else:
    df = X

print(f"✅ Étape 9 terminée - Dimensions: {df.shape}")

#===========================================
# SAUVEGARDE DANS PROCESSED/ (NOUVELLE ÉTAPE)
print("\n" + "="*50)
print("AJOUT: SAUVEGARDE DATASET NETTOYÉ")
print("="*50)

# PROCESSED_DIR déjà défini au début du fichier
output_path = os.path.join(PROCESSED_DIR, "eda_results.csv")
df_clean.to_csv(output_path, index=False)

print(f"   📁 Dossier: {PROCESSED_DIR}")
print(f"   💾 Dataset sauvegardé: {output_path}")
print(f"   Dimensions: {df_clean.shape}")
print(f"   ⚠️  Le dataset original dans raw/ est INTACT")

print("\n" + "="*50)
print("FIN DE L'EDA")
print("="*50)
