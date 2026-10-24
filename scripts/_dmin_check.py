import pandas as pd, numpy as np, json
from pathlib import Path
import bench_config as cfg
R = Path(__file__).resolve().parents[1] / "results"
df = pd.read_csv(R / "benchmark_raw.csv")
df = df[df.dataset.isin(cfg.ROSTER) & df.method.isin(cfg.MAIN_METHODS)].copy()

dmin = df.groupby(['dataset','method'])['min_train_distance'].mean().unstack()
mmd = df.groupby(['dataset','method'])['mmd2_test'].mean().unstack()
comp = [m for m in cfg.MAIN_METHODS if m != 'MomentAnchor']
print("=== per-dataset d_min: MA rank among 8, MA value, max competitor ===")
for ds in cfg.ROSTER:
    row = dmin.loc[ds]
    ma = row['MomentAnchor']
    rank = (row > ma).sum() + 1  # 1 = largest
    print(f"{ds:20s} MA_dmin={ma:8.3f} rank={rank}/8  maxComp={row[comp].max():8.3f}({row[comp].idxmax()[:8]})")
print("\n=== per-dataset MMD2: MA value, best, within-1std tie? ===")
for ds in cfg.ROSTER:
    sub = df[df.dataset==ds]
    ma_seed = sub[sub.method=='MomentAnchor']['mmd2_test']
    ma = ma_seed.mean()
    best = mmd.loc[ds].min(); bm = mmd.loc[ds].idxmin()
    ma_std = ma_seed.std()
    tie = ma - ma_std <= best
    print(f"{ds:20s} MA={ma:.2e}(std {ma_std:.1e}) best={best:.2e}({bm[:8]}) tie_within1std={tie}")

# how many datasets MA is best or tied on mmd2 within 1 std
print("\n=== certificate on roster ===")
ma = df[df.method=='MomentAnchor'].copy()
cert = ma['meta'].apply(lambda s: json.loads(s).get('certificate',{}))
ma['U']=cert.apply(lambda c:c.get('U_star',np.nan))
ma['rmarg']=cert.apply(lambda c:c.get('r_marg',np.nan))
ma['rmom']=cert.apply(lambda c:c.get('r_mom',np.nan))
ma['pool']=ma['meta'].apply(lambda s: json.loads(s).get('pool_size',np.nan))
ma['supp']=ma['meta'].apply(lambda s: json.loads(s).get('active_support',np.nan))
print(f"U*: {ma.U.min():.2g}..{ma.U.max():.2g} median {ma.U.median():.2g}")
print(f"r_mom: {ma.rmom.min():.2g}..{ma.rmom.max():.2g} median {ma.rmom.median():.2g}")
print(f"r_marg: {ma.rmarg.min():.2g}..{ma.rmarg.max():.2g} median {ma.rmarg.median():.2g}")
print(f"pool: mean {ma.pool.mean():.0f} range {ma.pool.min():.0f}..{ma.pool.max():.0f}")
print(f"active support: mean {ma.supp.mean():.0f}")
dsm = ma.groupby('dataset')[['U','mmd2_test']].mean()
print(f"corr(U*, test MMD2) over {len(dsm)} dataset means = {np.corrcoef(dsm.U,dsm.mmd2_test)[0,1]:.3f}")
