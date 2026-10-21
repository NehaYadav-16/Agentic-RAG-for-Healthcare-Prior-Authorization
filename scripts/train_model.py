"""Trains the denial-risk classifier and saves it to models/denial_model.joblib.

    python scripts/train_model.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paig.denial_model import save_model, train_denial_model  # noqa: E402

if __name__ == "__main__":
    claims_path = ROOT / "data" / "claims" / "claims_synthetic.csv"
    if not claims_path.exists():
        raise SystemExit("Run scripts/generate_data.py first to create the synthetic dataset.")

    df = pd.read_csv(claims_path)
    pipeline, metrics = train_denial_model(df)

    model_path = ROOT / "models" / "denial_model.joblib"
    save_model(pipeline, model_path)

    metrics_path = ROOT / "models" / "metrics.json"
    metrics_path.write_text(json.dumps(metrics.as_dict(), indent=2))

    print(f"Saved model to {model_path}")
    print(f"Test-set metrics: {json.dumps(metrics.as_dict(), indent=2)}")
