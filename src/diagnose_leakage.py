from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

for path in sorted(DATA_DIR.glob("*_features.csv")):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print('\n' + '='*60)
    print(path.name)
    if 'Signal' not in df.columns:
        print('  No Signal column')
        continue
    s = df['Signal']
    print('  Signal value counts:')
    print(s.value_counts(dropna=False))
    num = df.select_dtypes(include=['number'])
    corrs = num.corrwith(s).abs().sort_values(ascending=False)
    print('\n  Top correlations with Signal:')
    print(corrs.head(10))
    high = corrs[corrs > 0.95]
    if not high.empty:
        print('\n  Highly correlated (>0.95):')
        for col, val in high.items():
            equal = (num[col].fillna(9999999) == s.fillna(9999999)).all()
            print(f'   - {col}: corr={val:.4f}  exact_match={equal}')
    else:
        print('\n  No features with abs(corr) > 0.95')

print('\nDone')
