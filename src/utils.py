import pandas as pd
import joblib
import os

def save_data(df: pd.DataFrame, filepath: str):
    """Sauvegarde un DataFrame avec création automatique des dossiers"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"   💾 Sauvegardé : {filepath}")

def save_model(model, filepath: str):
    """Sauvegarde un modèle avec création automatique des dossiers"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"   💾 Modèle sauvegardé : {filepath}")

def load_model(filepath: str):
    """Charge un modèle sauvegardé"""
    return joblib.load(filepath)

def load_data(filepath: str) -> pd.DataFrame:
    """Charge des données CSV"""
    return pd.read_csv(filepath)