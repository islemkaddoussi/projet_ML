import pandas as pd
import numpy as np
import joblib
import argparse
import sys
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import load_model, save_data

def predict_on_test(output_path='data/predictions/predictions.csv'):
    """
    Prédiction sur les données de TEST (jamais vues par le modèle)
    """
    print("🔮 Prédiction sur données de TEST (validation réelle)")
    
    # Vérification fichiers
    required_files = [
        'data/train_test/X_test_pca.csv',
        'data/train_test/y_test.csv',
        'data/train_test/y_reg_test.csv',
        'data/train_test/X_test_scaled.csv',
        'models/churn_classifier.pkl',
        'models/monetary_regressor.pkl'
    ]
    
    for f in required_files:
        if not os.path.exists(f):
            print(f"❌ Fichier manquant : {f}")
            print("   Lancez : python src/preprocessing.py && python src/train_model.py")
            sys.exit(1)
    
    # Chargement données de TEST
    print("   Chargement X_test...")
    X_test = pd.read_csv('data/train_test/X_test_pca.csv')
    y_test_class = pd.read_csv('data/train_test/y_test.csv')['Churn']
    y_test_reg = pd.read_csv('data/train_test/y_reg_test.csv')['MonetaryTotal']
    X_test_scaled = pd.read_csv('data/train_test/X_test_scaled.csv')
    
    # Chargement modèles
    churn_model = load_model('models/churn_classifier.pkl')
    monetary_model = load_model('models/monetary_regressor.pkl')
    
    # ============================================================
    # PRÉDICTION
    # ============================================================
    print("   Prédiction en cours...")
    
    # Classification (sur PCA)
    churn_proba = churn_model.predict_proba(X_test)[:, 1]
    churn_pred = (churn_proba > 0.5).astype(int)
    
    # Régression (sur features scalées)
    monetary_pred = monetary_model.predict(X_test_scaled)
    
    # ============================================================
    # MÉTRIQUES (vraie évaluation)
    # ============================================================
    acc = accuracy_score(y_test_class, churn_pred)
    prec = precision_score(y_test_class, churn_pred, zero_division=0)
    rec = recall_score(y_test_class, churn_pred, zero_division=0)
    f1 = f1_score(y_test_class, churn_pred, zero_division=0)
    
    print(f"\n📊 Résultats sur TEST SET ({len(y_test_class)} clients) :")
    print(f"   Accuracy  : {acc:.3f}")
    print(f"   Precision : {prec:.3f}")
    print(f"   Recall    : {rec:.3f}")
    print(f"   F1-Score  : {f1:.3f}")
    
    # ============================================================
    # SAUVEGARDE
    # ============================================================
    results = pd.DataFrame({
        'Customer_Index': range(len(y_test_class)),
        'Churn_Probability': churn_proba.round(3),
        'Churn_Predicted': churn_pred,
        'Churn_Actual': y_test_class.values,
        'Prediction_Correct': churn_pred == y_test_class.values,
        'Risk_Level': pd.cut(churn_proba, 
                            bins=[0, 0.3, 0.6, 0.8, 1.0],
                            labels=['Faible', 'Moyen', 'Élevé', 'Critique']),
        'Monetary_Predicted': monetary_pred.round(2),
        'Monetary_Actual': y_test_reg.values.round(2)
    })
    
    save_data(results, output_path)
    
    # Résumé
    print(f"\n✅ Prédictions sauvegardées : {output_path}")
    print(f"   Churn détecté : {churn_pred.sum()}/{len(churn_pred)} ({churn_pred.mean()*100:.1f}%)")
    print(f"   Erreur moyenne Monetary : {np.abs(monetary_pred - y_test_reg).mean():.2f}£")
    
    return results


def predict_sample_from_raw(n=10, output_path='data/predictions/predictions_new.csv'):
    """
    Prédiction sur un échantillon aléatoire du fichier brut (simulation nouveaux clients)
    """
    raw_path = 'data/raw/retail_customers_COMPLETE_CATEGORICAL.csv'
    
    if not os.path.exists(raw_path):
        print(f"❌ Fichier brut non trouvé : {raw_path}")
        sys.exit(1)
    
    print(f"📊 Création d'un échantillon de {n} nouveaux clients...")
    df_full = pd.read_csv(raw_path)
    
    # Échantillon aléatoire (stratifié si possible)
    sample = df_full.sample(n, random_state=42).reset_index(drop=True)
    sample_path = 'data/raw/sample_new_clients.csv'
    sample.to_csv(sample_path, index=False)
    print(f"   Échantillon sauvegardé : {sample_path}")
    
    print("\n⚠️  NOTE : Pour prédire sur de vraies nouvelles données,")
    print("    il faut refaire le preprocessing complet (preprocessing.py)")
    print("    avec ce fichier en entrée.")
    print("\n    Solution : utilisez --mode=test pour l'instant")


def main():
    parser = argparse.ArgumentParser(description='Prédiction Churn & Monetary')
    parser.add_argument('--mode', choices=['test', 'sample'], default='test',
                       help='test: données de test (validation), sample: échantillon brut')
    parser.add_argument('--output', default='data/predictions/predictions.csv',
                       help='Fichier de sortie')
    parser.add_argument('--n', type=int, default=10,
                       help='Nombre de clients pour mode sample')
    
    args = parser.parse_args()
    os.makedirs('data/predictions', exist_ok=True)
    
    if args.mode == 'test':
        predict_on_test(args.output)
    else:
        predict_sample_from_raw(args.n, args.output)


if __name__ == "__main__":
    main()