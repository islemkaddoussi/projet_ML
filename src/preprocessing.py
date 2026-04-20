import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
from utils import save_data

RAW_PATH = 'data/raw/retail_customers_COMPLETE_CATEGORICAL.csv'

def main():
    # Création des dossiers nécessaires
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/train_test', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    print("🚀 === PRÉTRAITEMENT SIMPLIFIÉ ET CORRIGÉ ===")
    
    # ============================================================
    # 1. CHARGEMENT & NETTOYAGE
    # ============================================================
    print("\n1️⃣ CHARGEMENT & NETTOYAGE")
    df = pd.read_csv(RAW_PATH)
    print(f"   Données brutes : {df.shape}")

    # Parsing date
    df['RegistDate'] = pd.to_datetime(df['RegistrationDate'], dayfirst=True, errors='coerce')
    df['RegYear'] = df['RegistDate'].dt.year
    df['RegMonth'] = df['RegistDate'].dt.month
    df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency']
    if 'CustomerTenure' in df.columns:
        df['TenureRatio'] = df['Recency'] / (df['CustomerTenure'] + 1)

    # Age → NaN
    df.loc[~df['Age'].between(18, 81), 'Age'] = np.nan
    
    # Suppression features inutiles
    df = df.drop(columns=['Newsletter', 'LastLoginIP', 'RegistrationDate', 'RegistDate'], errors='ignore')

    # ============================================================
    # ENCODAGE CATÉGORIEL
    # ============================================================
    
    # One-hot
    onehot_cols = ['CustomerType', 'FavoriteSeason', 'Region', 'Gender', 
                   'AccountStatus', 'ProdDiversity', 'WeekendPref']
    onehot_present = [c for c in onehot_cols if c in df.columns]
    print(f"   One-hot sur : {onehot_present}")
    
    if onehot_present:
        df = pd.get_dummies(df, columns=onehot_present, drop_first=True)

    # Ordinal avec NaN
    ordinal_mappings = {
        'SpendingCat': {'Low':0,'Medium':1,'High':2,'VIP':3},
        'LoyaltyLevel': {'Nouveau':0,'Jeune':1,'Établi':2,'Ancien':3,'Inconnu':np.nan},
        'ChurnRisk': {'Faible':0,'Moyen':1,'Élevé':2,'Critique':3},
        'AgeCategory': {'18-24':0,'25-34':1,'35-44':2,'45-54':3,'55-64':4,'65+':5,'Inconnu':np.nan},
        'BasketSize': {'Petit':0,'Moyen':1,'Grand':2,'Inconnu':np.nan}
    }
    
    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Suppression fuites
    leakage_cols = ['ChurnRiskCategory', 'ChurnRisk']
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors='ignore')

    # Conversion types non-numériques résiduels
    non_numeric = df.select_dtypes(exclude=[np.number, 'bool']).columns.tolist()
    non_numeric = [c for c in non_numeric if c not in ['Churn', 'CustomerID']]
    
    if non_numeric:
        print(f"   ⚠️  Non-numériques : {non_numeric}")
        for col in non_numeric:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"      → {col} converti")
            except:
                print(f"      → {col} SUPPRIMÉ")
                df = df.drop(columns=[col])

    print(f"   Types finaux : {df.dtypes.value_counts().to_dict()}")

    # ============================================================
    # 2. SPLIT
    # ============================================================
    print("\n2️⃣ SPLIT TRAIN/TEST")
    
    y_class = df['Churn']
    y_reg = df['MonetaryTotal']
    X = df.drop(columns=['Churn', 'CustomerID', 'MonetaryTotal'], errors='ignore')
    
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        X, y_class, y_reg,
        test_size=0.2,
        random_state=42,
        stratify=y_class
    )
    print(f"   Train : {X_train.shape} | Test : {X_test.shape}")

    # ============================================================
    # 3. POST-SPLIT
    # ============================================================
    print("\n3️⃣ TRAITEMENT POST-SPLIT")

    # Target Encoding Country
    if 'Country' in X_train.columns:
        print("   → Target Encoding Country...")
        target_mean = pd.concat([X_train['Country'], y_train_class], axis=1).groupby('Country')['Churn'].mean()
        
        X_train['Country_TargetEnc'] = X_train['Country'].map(target_mean)
        X_test['Country_TargetEnc'] = X_test['Country'].map(target_mean).fillna(y_train_class.mean())
        
        X_train = X_train.drop(columns=['Country'])
        X_test = X_test.drop(columns=['Country'])

    # IsolationForest
    print("   → IsolationForest sur train...")
    iso_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    iso = IsolationForest(contamination=0.06, random_state=42)
    mask = iso.fit_predict(X_train[iso_features].fillna(0)) == 1
    
    X_train_clean = X_train.loc[mask, iso_features].reset_index(drop=True)
    y_train_class_clean = y_train_class[mask].reset_index(drop=True)
    y_train_reg_clean = y_train_reg[mask].reset_index(drop=True)
    
    print(f"   Gardés : {X_train_clean.shape}")

    # Suppression colonnes constantes
    print("   → Suppression colonnes constantes...")
    
    constant_cols = X_train_clean.columns[X_train_clean.nunique() <= 1].tolist()
    if constant_cols:
        print(f"   ⚠️  Colonnes constantes supprimées : {constant_cols}")
        X_train_clean = X_train_clean.drop(columns=constant_cols)
        X_test = X_test.drop(columns=constant_cols, errors='ignore')

    all_nan_cols = X_train_clean.columns[X_train_clean.isna().all()].tolist()
    if all_nan_cols:
        print(f"   ⚠️  Colonnes 100% NaN supprimées : {all_nan_cols}")
        X_train_clean = X_train_clean.drop(columns=all_nan_cols)
        X_test = X_test.drop(columns=all_nan_cols, errors='ignore')

    print(f"   Shape après nettoyage : {X_train_clean.shape}")

    # KNN Imputer
    print("   → KNN Imputer...")
    imputer = KNNImputer(n_neighbors=5)
    
    X_test_aligned = X_test[X_train_clean.columns].copy()
    
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train_clean), 
        columns=X_train_clean.columns
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test_aligned), 
        columns=X_train_clean.columns
    )
    
    joblib.dump(imputer, 'models/imputer.joblib')

    # VIF
    print("   → Calcul VIF...")
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_train_imp.columns
    vif_data["VIF"] = [variance_inflation_factor(X_train_imp.values, i) 
                       for i in range(X_train_imp.shape[1])]
    
    # CORRECTION : to_drop (pas To_drop)
    to_drop = vif_data[vif_data['VIF'] > 10]['feature'].tolist()
    if to_drop:
        print(f"   ⚠️  Supprimés (VIF>10) : {to_drop}")
        X_train_imp = X_train_imp.drop(columns=to_drop)
        X_test_imp = X_test_imp.drop(columns=to_drop)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    joblib.dump(scaler, 'models/scaler.joblib')

    # PCA
    pca = PCA(n_components=10, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    joblib.dump(pca, 'models/pca.joblib')
    print(f"   PCA variance : {pca.explained_variance_ratio_.sum():.1%}")

    # ============================================================
    # 4. SAUVEGARDE
    # ============================================================
    print("\n4️⃣ SAUVEGARDE")
    
    save_data(pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(10)]), 'data/train_test/X_train_pca.csv')
    save_data(pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(10)]), 'data/train_test/X_test_pca.csv')
    
    save_data(pd.DataFrame({'Churn': y_train_class_clean}), 'data/train_test/y_train.csv')
    save_data(pd.DataFrame({'Churn': y_test_class}), 'data/train_test/y_test.csv')
    save_data(pd.DataFrame({'MonetaryTotal': y_train_reg_clean}), 'data/train_test/y_reg_train.csv')
    save_data(pd.DataFrame({'MonetaryTotal': y_test_reg}), 'data/train_test/y_reg_test.csv')
    
    save_data(pd.DataFrame(X_train_scaled, columns=X_train_imp.columns), 'data/train_test/X_train_scaled.csv')
    save_data(pd.DataFrame(X_test_scaled, columns=X_train_imp.columns), 'data/train_test/X_test_scaled.csv')
    # AJOUT : Sauvegarde des données nettoyées complètes pour predict.py
    train_clean = X_train_imp.copy()
    train_clean['Churn'] = y_train_class_clean.values
    train_clean['MonetaryTotal'] = y_train_reg_clean.values
    save_data(train_clean, 'data/processed/cleaned_data.csv')  # <-- AJOUTER CECI

    print("\n✅ PRÉTRAITEMENT TERMINÉ")

if __name__ == "__main__":
    main()