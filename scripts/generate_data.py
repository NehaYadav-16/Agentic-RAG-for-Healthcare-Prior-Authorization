"""Regenerates the synthetic claims dataset. Run from the project root:

    python scripts/generate_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paig.data_gen import generate_synthetic_claims  # noqa: E402

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "data" / "claims"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_claims()
    out_path = out_dir / "claims_synthetic.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Denial rate: {df['denied'].mean():.1%}")
