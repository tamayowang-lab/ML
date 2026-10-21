# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:47:47 2026

@author: ttq766
"""

"""
Ordinal logistic regression for W, N2 and N3 (decreasing E/I ratio) from region x band SampEn.

One model for all three stages instead of pairwise comparisons:
  * Stage order W = 0 < N2 = 1 < N3 = 2, i.e. a higher class means a lower E/I ratio.
  * Proportional-odds model: P(stage <= k | x) = sigmoid(theta_k - x @ w).
    A positive coefficient means that a higher feature value goes with deeper sleep (lower E/I).
  * The single latent score x @ w is the candidate E/I-tracking biomarker. Pairwise AUCs
    (W vs N2, N2 vs N3, W vs N3) are computed from this one score.
  * Nested CV, grouped by subject, with imputation, scaling, RFE, number of features and C
    all fitted on training data only (same logic as sleep_stage_classification.py).
  * Tuning criterion: Spearman correlation between the latent score and the stage.
"""
import numpy as np
import pandas as pd
from joblib import delayed
from tqdm import tqdm
import os
import re
import pickle
import warnings
from pathlib import Path
import mne

from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# ----------------------------------------------------------------------------
def channel_region(name):
    """'Fp1' -> 'Frontal', 'T8' -> 'Temporal_R', 'M1' -> None (excluded)."""
    m = re.match(r'^(.{1,2})(.)$', name)
    if not m or m.group(1) not in prefix_to_region:
        return None
 
    region, _ = prefix_to_region[m.group(1)], m.group(2)
    return region


def build_region_index(names):
    if names is None or len(names) != n_channels:
        raise ValueError(f'Set channel_names to the {n_channels} channel names '
                         f'in file row order.')
    regions = {}
    for i, n in enumerate(names):
        r = channel_region(n)
        if r is None:
            print(f'  Note: channel {n!r} not assigned to any region (excluded)')
            continue
        regions.setdefault(r, []).append(i)
    return {r: np.array(regions[r]) for r in sorted(regions)}


load_folder = r'D:\Sleep_data\Preprocessing\Clean_final'
file_names = [f for f in os.listdir(load_folder) if f.endswith('raw.fif')]
file_names.sort()
file_name = file_names[0]
file_path = os.path.join(load_folder, file_name)
raw_data = mne.io.read_raw_fif(file_path, preload=True)
channel_names = raw_data.info['ch_names']

########
bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']     # folder names
n_channels = len(channel_names)
extensions = {'.mrk'}
mat_key = None          # variable name inside .mat files; none = auto if only one

# region definition from the electrode-name prefix (edit to taste).
prefix_to_region = {
    'Fp': 'frontal', 'AF': 'frontal', 'F': 'frontal',
    'FC': 'central', 'C': 'central', 'CP': 'central',
    'P': 'parietal',
    'PO': 'occipital', 'O': 'occipital',
    'FT': 'temporal', 'T': 'temporal', 'TP': 'temporal',   # split l/r below
}

# stage labels in the output table keep your own naming (W, N2, N3, R)
stage_map = {'wake': 'W', 'n1': 'N1','n2': 'N2', 'n3': 'N3', 'r': 'R', 'rem': 'R'}

# Region definition from the electrode-name prefix (edit to taste).
prefix_to_region = {
    'Fp': 'Frontal', 'AF': 'Frontal', 'F': 'Frontal',
    'FC': 'Central', 'C': 'Central', 'CP': 'Central',
    'P': 'Parietal',
    'PO': 'Occipital', 'O': 'Occipital',
    'FT': 'Temporal', 'T': 'Temporal', 'TP': 'Temporal',   # Split into L/R below
}

# Filename = subject id followed by the stage name (W, N2, N3 or R)
file_re = re.compile(
    r'^(?P<subj>.+?)[_\-\s]+(?P<stage>N1|N2|N3|W|R)(?:_\d+)*$'
)
region_idx = build_region_index(channel_names)

#%%
base_dir = Path(r'D:\Sleep_data\Markers\SampEn\Dim2\No_sp_sw')

store = {}                          # (subject, stage) -> {band: regions x epochs}
for band in bands:
    for f in sorted((Path(base_dir) / band).glob('*.mrk')):
        m = file_re.match(f.stem)
        if not m:
            print(f'  Skipped (name not understood): {f.name}')
            continue
        subj, stage = m['subj'], m['stage']

        with open(f, 'rb') as fh:
            arr = np.asarray(pickle.load(fh), dtype=float)
        if arr.shape[0] != n_channels:      # Fix channels x epochs orientation
            arr = arr.T
        arr[~np.isfinite(arr)] = np.nan     # SampEn can return inf/NaN

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            reg = np.vstack([np.nanmean(arr[idx], axis=0) for idx in region_idx.values()])
        store.setdefault((subj, stage), {})[band] = reg

regions = list(region_idx)
frames = []
for (subj, stage), per_band in sorted(store.items()):
    missing = [b for b in bands if b not in per_band]
    if missing:
        print(f'  Skipped {subj} {stage}: missing bands {missing}')
        continue
    n_ep = {b: v.shape[1] for b, v in per_band.items()}
    if len(set(n_ep.values())) > 1:
        raise ValueError(f'{subj} {stage}: epoch counts differ across bands {n_ep}')
    n = next(iter(n_ep.values()))
    feats = {f'SampEn_{b}_{r}': per_band[b][i]
             for b in bands for i, r in enumerate(regions)}
    df = pd.DataFrame(feats)
    df.insert(0, 'epoch', np.arange(n))
    df.insert(0, 'stage', stage)
    df.insert(0, 'original_id', subj)
    frames.append(df)

data_sampen= pd.concat(frames, ignore_index=True)

print(data_sampen.groupby(['stage']).size())
print(data_sampen.groupby(['original_id', 'stage']).size().unstack())
data_sampen.to_csv('sampen_features_epochs.csv', index=False)
print('Saved sampen_features_epochs.csv', data_sampen.shape)

#%% DFA
base_dir = Path(r'D:\Sleep_data\Markers\DFA')

store = {}                          # (subject, stage) -> {band: regions x epochs}
for band in bands:
    for f in sorted((Path(base_dir) / band).glob('*.mrk')):
        m = file_re.match(f.stem)
        if not m:
            print(f'  Skipped (name not understood): {f.name}')
            continue
        subj, stage = m['subj'], m['stage']

        with open(f, 'rb') as fh:
            arr = np.asarray(pickle.load(fh), dtype=float)
        if arr.shape[0] != n_channels:      # Fix channels x epochs orientation
            arr = arr.T
        arr[~np.isfinite(arr)] = np.nan     # SampEn can return inf/NaN

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            reg = np.vstack([np.nanmean(arr[idx], axis=0) for idx in region_idx.values()])
        store.setdefault((subj, stage), {})[band] = reg

regions = list(region_idx)
frames = []
for (subj, stage), per_band in sorted(store.items()):
    missing = [b for b in bands if b not in per_band]
    if missing:
        print(f'  Skipped {subj} {stage}: missing bands {missing}')
        continue
    n_ep = {b: v.shape[1] for b, v in per_band.items()}
    if len(set(n_ep.values())) > 1:
        raise ValueError(f'{subj} {stage}: epoch counts differ across bands {n_ep}')
    n = next(iter(n_ep.values()))
    feats = {f'SampEn_{b}_{r}': per_band[b][i]
             for b in bands for i, r in enumerate(regions)}
    df = pd.DataFrame(feats)
    df.insert(0, 'epoch', np.arange(n))
    df.insert(0, 'stage', stage)
    df.insert(0, 'original_id', subj)
    frames.append(df)

data_dfa = pd.concat(frames, ignore_index=True)

print(data_dfa.groupby(['stage']).size())
print(data_dfa.groupby(['original_id', 'stage']).size().unstack())
data_dfa.to_csv('sampen_features_epochs.csv', index=False)
print('Saved dfa_epochs.csv', data_dfa.shape)

#%% fooof

# base_dir = Path(r'D:\Sleep_data\Markers\Slope_1_45')

# store = {}                          # (subject, stage) -> {band: regions x epochs}

# for f in sorted((Path(base_dir)).glob('*.mrk')):
#     m = file_re.match(f.stem)
#     if not m:
#         print(f'  Skipped (name not understood): {f.name}')
#         continue
#     subj, stage = m['subj'], m['stage']

#     with open(f, 'rb') as fh:
#         arr = np.asarray(pickle.load(fh), dtype=float)
#     if arr.shape[0] != n_channels:      # Fix channels x epochs orientation
#         arr = arr.T
#     arr[~np.isfinite(arr)] = np.nan     # SampEn can return inf/NaN

#     with warnings.catch_warnings():
#         warnings.simplefilter('ignore', category=RuntimeWarning)
#         reg = np.vstack([np.nanmean(arr[idx], axis=0) for idx in region_idx.values()])
#     store.setdefault((subj, stage), {})[band] = reg

# regions = list(region_idx)
# frames = []
# for (subj, stage), per_band in sorted(store.items()):
#     missing = [b for b in bands if b not in per_band]
#     if missing:
#         print(f'  Skipped {subj} {stage}: missing bands {missing}')
#         continue
#     n_ep = {b: v.shape[1] for b, v in per_band.items()}
#     if len(set(n_ep.values())) > 1:
#         raise ValueError(f'{subj} {stage}: epoch counts differ across bands {n_ep}')
#     n = next(iter(n_ep.values()))
#     feats = {f'SampEn_{b}_{r}': per_band[b][i]
#              for b in bands for i, r in enumerate(regions)}
#     df = pd.DataFrame(feats)
#     df.insert(0, 'epoch', np.arange(n))
#     df.insert(0, 'stage', stage)
#     df.insert(0, 'original_id', subj)
#     frames.append(df)

# data_slope = pd.concat(frames, ignore_index=True)


# print(data_slope.groupby(['stage']).size())
# print(data_slope.groupby(['original_id', 'stage']).size().unstack())
# data_slope.to_csv('sampen_features_epochs.csv', index=False)
# print('Saved 1_f_features_epochs.csv', data_slope.shape)
#%%
data = pd.merge(data_sampen, data_dfa, on=['original_id', 'stage', 'epoch'], how='inner')
#from ML import zscore_within_subject, impute_scale, run_jobs

settings = dict(
    outer_folds=5,
    inner_folds=5,
    repeats=10,
    seed=42,
    n_jobs=10,                           # 1 = single core, -1 = all cores, or any number of cores
    max_epochs_per_subject_stage=150,   # None uses all epochs
    zscore_within_subject=True,         # Label-free removal of between-subject offsets
)
stage_order = ['W', 'N1', 'N2', 'N3']         # Decreasing E/I ratio
c_grid = np.logspace(-3, 1, 5)


class OrdinalLogit:
    """Proportional-odds logistic regression with an L2 penalty.

    Objective: mean weighted negative log-likelihood + ||w||^2 / (2 * c * n),
    so c has the same meaning as C in sklearn's LogisticRegression.
    """

    def __init__(self, c=1.0, max_iter=500):
        self.c = c
        self.max_iter = max_iter

    @staticmethod
    def _unpack(params, p):
        # Thresholds are kept increasing: theta_0 = a0, theta_k = theta_(k-1) + exp(d_k)
        w, a0, d = params[:p], params[p], params[p + 1:]
        return w, np.cumsum(np.concatenate([[a0], np.exp(d)]))

    def _objective(self, params, x, y, s):
        n, p = x.shape
        w, theta = self._unpack(params, p)
        d = params[p + 1:]
        idx = np.arange(n)

        f = expit(theta[None, :] - (x @ w)[:, None])                  # n x (K-1), F_k
        fcum = np.hstack([np.zeros((n, 1)), f, np.ones((n, 1))])      # Columns: F_-1 ... F_(K-1)
        prob = np.clip(fcum[idx, y + 1] - fcum[idx, y], 1e-12, None)  # P(y) = F_y - F_(y-1)

        value = -np.mean(s * np.log(prob)) + w @ w / (2 * self.c * n)

        g = f * (1 - f)                                               # dF_k / dtheta_k
        gcum = np.hstack([np.zeros((n, 1)), g, np.zeros((n, 1))])
        r = s / (prob * n)
        grad_w = x.T @ (r * (gcum[idx, y + 1] - gcum[idx, y])) + w / (self.c * n)

        k = f.shape[1]
        grad_theta = np.array([-(np.sum(r[y == j] * g[y == j, j])
                                 - np.sum(r[y == j + 1] * g[y == j + 1, j])) for j in range(k)])
        rev = np.cumsum(grad_theta[::-1])[::-1]                       # rev[j] = sum of grad_theta[k >= j]
        grad = np.concatenate([grad_w, [rev[0]], np.exp(d) * rev[1:]])
        return value, grad

    def fit(self, x, y, sample_weight=None):
        self.classes_ = np.unique(y)
        y = np.searchsorted(self.classes_, y)
        n, p = x.shape
        k = len(self.classes_)
        s = np.ones(n) if sample_weight is None else sample_weight * n / np.sum(sample_weight)

        # Start from the (weighted) class proportions
        props = np.array([s[y == j].sum() for j in range(k)]) / s.sum()
        cum = np.clip(np.cumsum(props)[:-1], 1e-3, 1 - 1e-3)
        theta0 = np.log(cum / (1 - cum))
        init = np.concatenate([np.zeros(p), [theta0[0]],
                               np.log(np.clip(np.diff(theta0), 1e-3, None))])

        res = minimize(self._objective, init, args=(x, y, s), jac=True,
                       method='L-BFGS-B', options={'maxiter': self.max_iter})
        self.coef_, self.theta_ = self._unpack(res.x, p)
        return self

    def decision_function(self, x):
        return x @ self.coef_

    def predict_proba(self, x):
        f = expit(self.theta_[None, :] - self.decision_function(x)[:, None])
        n = len(f)
        return np.diff(np.hstack([np.zeros((n, 1)), f, np.ones((n, 1))]), axis=1)

    def predict(self, x):
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]


def rfe_rank(x, y, w):
    """RFE ranking (1 = most important): repeatedly drop the feature with the smallest |coef|.
    The top n features are those with rank <= n."""
    remaining = list(range(x.shape[1]))
    rank = np.zeros(x.shape[1], dtype=int)
    while len(remaining) > 1:
        model = OrdinalLogit(c=1.0).fit(x[:, remaining], y, w)
        worst = int(np.argmin(np.abs(model.coef_)))
        rank[remaining.pop(worst)] = len(remaining) + 1
    rank[remaining[0]] = 1
    return rank


def make_set(df, feat_cols, cap, seed):
    d = df[df['stage'].isin(stage_order)]
    if cap:
        parts = [g.sample(min(len(g), cap), random_state=seed)
                 for _, g in d.groupby(['original_id', 'stage'])]
        d = pd.concat(parts)
    d = d.reset_index(drop=True)
    y = d['stage'].map({st: i for i, st in enumerate(stage_order)}).values
    return d[feat_cols].values, y, d['original_id'].values


def run_fold(tr, te, x, y, groups, seed, inner_folds):
    n_feat = x.shape[1]
    n_grid = sorted({k for k in (2, 3, 4, 6, 8, 12, 16, 20, 25) if k < n_feat} | {n_feat})
    x_tr_all, y_tr_all, g_tr_all = x[tr], y[tr], groups[tr]

    # Inner CV: one RFE ranking per inner fold, reused for every n and every C
    inner = StratifiedGroupKFold(inner_folds, shuffle=True, random_state=seed)
    scores = np.full((inner_folds, len(n_grid), len(c_grid)), np.nan)
    for k, (ia, ib) in enumerate(inner.split(x_tr_all, y_tr_all, g_tr_all)):
        ya, yb = y_tr_all[ia], y_tr_all[ib]
        if len(np.unique(ya)) < len(stage_order) or len(np.unique(yb)) < 2:
            continue
        xa, xb = impute_scale(x_tr_all[ia], x_tr_all[ib])
        wa = compute_sample_weight('balanced', ya)
        rank = rfe_rank(xa, ya, wa)
        for i, n in enumerate(n_grid):
            cols = rank <= n
            for j, c in enumerate(c_grid):
                model = OrdinalLogit(c).fit(xa[:, cols], ya, wa)
                scores[k, i, j] = spearmanr(model.decision_function(xb[:, cols]), yb)[0]

    mean_scores = np.nanmean(scores, axis=0)
    i, j = np.unravel_index(np.nanargmax(mean_scores), mean_scores.shape)
    best_n, best_c = n_grid[i], c_grid[j]

    # Final model on all outer-training data
    x_tr, x_te = impute_scale(x_tr_all, x[te])
    w = compute_sample_weight('balanced', y_tr_all)
    cols = rfe_rank(x_tr, y_tr_all, w) <= best_n
    model = OrdinalLogit(best_c).fit(x_tr[:, cols], y_tr_all, w)

    yt = y[te]
    z = model.decision_function(x_te[:, cols])
    pred = model.predict(x_te[:, cols])

    out = dict(rho=spearmanr(z, yt)[0],
               ba=balanced_accuracy_score(yt, pred),
               mae=float(np.mean(np.abs(pred - yt))),
               n_feat=int(cols.sum()), c=best_c)
    for a, b in ((0, 1), (1, 2), (0, 2)):
        mask = np.isin(yt, [a, b])
        out[f'auc_{stage_order[a]}_vs_{stage_order[b]}'] = (
            roc_auc_score(yt[mask] == b, z[mask]) if len(np.unique(yt[mask])) == 2 else np.nan)
    coef_full = np.zeros(n_feat)
    coef_full[cols] = model.coef_
    out['coef'] = coef_full
    return out


def nested_cv(x, y, groups, s, bar):
    fold_results = []
    for r in range(s['repeats']):
        seed = s['seed'] + r
        outer = StratifiedGroupKFold(s['outer_folds'], shuffle=True, random_state=seed)
        jobs = [delayed(run_fold)(tr, te, x, y, groups, seed, s['inner_folds'])
                for tr, te in outer.split(x, y, groups)]
        for d in run_jobs(jobs, s['n_jobs'], bar):
            d['repeat'] = r
            fold_results.append(d)
    return fold_results


def summarise(fold_results, feat_cols):
    metrics = [k for k in fold_results[0] if k not in ('coef', 'repeat')]
    df = pd.DataFrame([{k: d[k] for k in metrics + ['repeat']} for d in fold_results])
    per_repeat = df.groupby('repeat')[metrics].mean()

    coefs = np.vstack([d['coef'] for d in fold_results])
    selected = coefs != 0
    n_sel = selected.sum(axis=0)
    pos = (coefs > 0).sum(axis=0)
    neg = (coefs < 0).sum(axis=0)
    importance = pd.DataFrame({
        'selection_freq': selected.mean(axis=0),
        'mean_coef': coefs.mean(axis=0),            # Zeros where not selected
        'sd_coef': coefs.std(axis=0),
        'sign_consistency': np.where(n_sel > 0, np.maximum(pos, neg) / np.maximum(n_sel, 1), np.nan),
        'mean_abs_coef': np.abs(coefs).mean(axis=0),
    }, index=feat_cols).sort_values('mean_abs_coef', ascending=False)

    parts = pd.DataFrame([n.split('_', 2) for n in importance.index],
                         columns=['prefix', 'band', 'region'], index=importance.index)
    by_band = importance['mean_abs_coef'].groupby(parts['band']).sum()
    by_region = importance['mean_abs_coef'].groupby(parts['region']).sum()
    return per_repeat, importance, (by_band / by_band.sum()).sort_values(ascending=False), \
        (by_region / by_region.sum()).sort_values(ascending=False)


s = settings

data = data[data['stage'].isin(stage_order)]
feat_cols = [c for c in data.columns if c.startswith('SampEn_')]
if s['zscore_within_subject']:
    data = zscore_within_subject(data, feat_cols)

x, y, groups = make_set(data, feat_cols, s['max_epochs_per_subject_stage'], s['seed'])
print(f'{len(y)} epochs, {len(np.unique(groups))} subjects, '
      f'stages {stage_order}, counts {np.bincount(y).tolist()}')

bar = tqdm(total=s['repeats'] * s['outer_folds'], unit='fold', desc='Ordinal')
fold_results = nested_cv(x, y, groups, s, bar)
bar.close()

per_repeat, importance, by_band, by_region = summarise(fold_results, feat_cols)
print('\n=== Performance (mean +- SD across repeats) ===')
for k in per_repeat.columns:
    print(f'  {k:>16s}: {per_repeat[k].mean():.3f} +- {per_repeat[k].std():.3f}')
print('\n=== Top features ===')
print(importance.head(10).round(3).to_string())
print('\n=== Share of importance by band ===')
print(by_band.round(3).to_string())
print('\n=== Share of importance by region ===')
print(by_region.round(3).to_string())

per_repeat.to_csv('ordinal_performance.csv')
importance.to_csv('ordinal_feature_importance.csv')


