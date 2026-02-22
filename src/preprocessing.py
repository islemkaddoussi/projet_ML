import pandas as pd
import numpy as np

import os

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

#============================
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

#============================
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

#============================
# ÉTAPE 6: MULTICOLINÉARITÉ

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