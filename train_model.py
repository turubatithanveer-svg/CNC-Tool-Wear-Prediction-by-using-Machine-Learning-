"""
=============================================================================
train_model.py — CLI Training Runner
=============================================================================
Run this script to train all models from the command line:

    python train_model.py

Outputs models to the models/ directory.
=============================================================================
"""

import sys
import os

# Ensure project root is in path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.preprocessing import load_dataset, preprocess, save_preprocessors
from src.train import run_training_pipeline

DATASET_PATH = os.path.join(ROOT, 'dataset', 'ai4i2020.csv')
MODEL_DIR    = os.path.join(ROOT, 'models')


def main():
    print("=" * 70)
    print("  CNC PREDICTIVE MAINTENANCE — MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load
    print("\n📂 Step 1: Loading dataset…")
    df = load_dataset(DATASET_PATH)

    # 2. Preprocess
    print("\n🔧 Step 2: Preprocessing…")
    data_dict = preprocess(df)

    # 3. Save preprocessors
    print("\n💾 Step 3: Saving preprocessors…")
    save_preprocessors(data_dict['scaler'], data_dict['encoder'], MODEL_DIR)

    # 4. Train
    print("\n🤖 Step 4: Training all models…")
    results = run_training_pipeline(data_dict, MODEL_DIR)

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print(f"   Best Model : {results['best']['name']}")
    print(f"   Accuracy   : {results['best']['accuracy']*100:.2f}%")
    print(f"   F1 Score   : {results['best']['f1']*100:.2f}%")
    print(f"   ROC AUC    : {results['best']['roc_auc']*100:.2f}%")
    print(f"\n   Models saved to: {MODEL_DIR}/")
    print("=" * 70)
    print("\n🚀 Now run:  streamlit run app.py")


if __name__ == '__main__':
    main()
