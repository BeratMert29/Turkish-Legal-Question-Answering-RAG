"""Validate raw dataset — read-only, no output files."""
import pandas as pd
import sys
from pathlib import Path
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.append(_project_root)
import config

def main():
    print(f"Loading: {config.RAW_DATA_PATH}")
    df = pd.read_csv(config.RAW_DATA_PATH, dtype=str, keep_default_na=False, low_memory=False)

    print("\n=== Dataset Summary ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\n=== Split Distribution ===")
    print(df['split'].value_counts().to_string())

    print("\n=== Null counts per split ===")
    for split_name in ['kaggle', 'train', 'test']:
        sub = df[df['split'] == split_name]
        nulls = (sub == '').sum()
        print(f"{split_name}: {len(sub)} rows, empty context={nulls.get('context', 0)}")

    print("\n=== Context length stats (kaggle rows) ===")
    kaggle = df[df['split'] == 'kaggle']
    ctx_lens = kaggle['context'].str.len()
    print(ctx_lens.describe().to_string())

    print("\n✓ Validation complete")

if __name__ == '__main__':
    main()
