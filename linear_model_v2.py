from __future__ import annotations
import argparse, logging, os, warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# CONFIGURATION & LOGGING
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# DATA LOADING & ALIGNMENT
# ---------------------------------------------------------------------------
class DataLoader:
    def __init__(self, x_path: Path, y_path: Path):
        self.x_df = self._read(x_path)
        self.y_df = self._read(y_path)

    @staticmethod
    def _read(p: Path) -> pd.DataFrame:
        if p.suffix.lower() in {'.parquet', '.pq'}:
            return pd.read_parquet(p)
        if p.suffix.lower() == '.csv':
            return pd.read_csv(p, index_col=0, parse_dates=True)
        raise ValueError(f"Unsupported file type {p.suffix}")

    def load_panel(self, target_col: str, shift_target: bool = True) -> pd.DataFrame:
        df = self.x_df.join(
            self.y_df.rename(columns={self.y_df.columns[0]: target_col}), how='inner'
        )
        if shift_target:
            df[target_col] = df[target_col].shift(-1)
        df = df.ffill().dropna()
        logging.info("Panel loaded & aligned: %d obs × %d cols", *df.shape)
        return df

# ---------------------------------------------------------------------------
# FEATURE ENGINEERING TRANSFORMERS
# ---------------------------------------------------------------------------
class StaticZ(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None): return self
    def transform(self, X): return (X - X.mean()) / X.std()

class RollingZ(TransformerMixin, BaseEstimator):
    def __init__(self, window: int = 36): self.window = window
    def fit(self, X, y=None): return self
    def transform(self, X):
        mu = X.rolling(self.window).mean()
        sd = X.rolling(self.window).std()
        return (X - mu) / sd

class Lagger(TransformerMixin, BaseEstimator):
    def __init__(self, lags: int = 3): self.lags = lags
    def fit(self, X, y=None): return self
    def transform(self, X):
        return pd.concat(
            [X.shift(lag).rename(lambda c: f"{c}_lag{lag}", axis=1)
             for lag in range(1, self.lags+1)], axis=1
        )

class Delta(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None): return self
    def transform(self, X): return X.diff()

# ---------------------------------------------------------------------------
# PIPELINE BUILDER
# ---------------------------------------------------------------------------
def build_feature_pipeline(
    mode: str,
    use_lags: bool,
    use_delta: bool,
    base_cols: List[str]
) -> Pipeline:
    transformers = []
    if mode == 'static_z':
        transformers.append(('static_z', StaticZ(), base_cols))
    elif mode == 'rolling_z':
        transformers.append(('rolling_z', RollingZ(window=36), base_cols))
    if use_lags:
        transformers.append(('lags', Lagger(lags=3), base_cols))
    if use_delta:
        transformers.append(('delta', Delta(), base_cols))
    if transformers:
        ct = ColumnTransformer(transformers, remainder='drop')
        return Pipeline([
            ('features', ct),
            ('scale', StandardScaler(with_mean=False))
        ])
    # raw features only
    return Pipeline([('scale', StandardScaler())])

# ---------------------------------------------------------------------------
# MODEL & HYPERPARAMETER GRID
# ---------------------------------------------------------------------------
LINEAR_MODELS = {
    'OLS': (LinearRegression(), {}),
    'Ridge': (Ridge(random_state=SEED), {'alpha': [0.01, 0.1, 1.0, 10.0]}),
    'Lasso': (Lasso(random_state=SEED), {'alpha': [0.001, 0.01, 0.1, 1.0]}),
    'ElasticNet': (
        ElasticNet(random_state=SEED),
        {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]}
    ),
    'BayesianRidge': (BayesianRidge(), {}),
    'PCR': (
        Pipeline([('pca', PCA()), ('lr', LinearRegression())]),
        {'pca__n_components': [1, 2, 5, 10, 0.95]}
    ),
    'PLS': (
        PLSRegression(),
        {'n_components': [1, 2, 5, 10]}
    )
}

# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

# ---------------------------------------------------------------------------
# WALK-FORWARD EVALUATION
# ---------------------------------------------------------------------------
def evaluate_all_combinations(
    df: pd.DataFrame,
    target_col: str,
    modes: List[str],
    train_periods: int,
    test_periods: int,
    window_type: str
) -> pd.DataFrame:
    records = []
    combos = [(m, l, d) for m in modes for l in [False, True] for d in [False, True]]
    n = len(df)

    for mode, use_lags, use_delta in combos:
        for start in range(train_periods, n - test_periods + 1, test_periods):
            # define train/test index ranges
            if window_type == 'expanding':
                idx_tr = slice(0, start)
            else:
                idx_tr = slice(start - train_periods, start)
            idx_te = slice(start, start + test_periods)

            train = df.iloc[idx_tr]
            test = df.iloc[idx_te]
            X_tr, y_tr = train.drop(columns=[target_col]), train[target_col]
            X_te, y_te = test.drop(columns=[target_col]), test[target_col]

            # build & apply feature pipeline
            pipe = build_feature_pipeline(mode, use_lags, use_delta, list(X_tr.columns))
            X_tr_fe = pd.DataFrame(pipe.fit_transform(X_tr), index=X_tr.index).dropna()
            y_tr_aligned = y_tr.reindex(X_tr_fe.index)
            X_te_fe = pd.DataFrame(pipe.transform(X_te), index=X_te.index).dropna()
            y_te_aligned = y_te.reindex(X_te_fe.index)

            # skip windows without data after FE
            if X_tr_fe.empty or X_te_fe.empty:
                continue

            for name, (estimator, grid) in LINEAR_MODELS.items():
                # dynamic grid filtering for PCA/PLS
                grid_use = {}
                for param, vals in grid.items():
                    filtered = [v for v in vals if not isinstance(v, int) or v <= X_tr_fe.shape[1]]
                    if filtered:
                        grid_use[param] = filtered
                # model selection
                if grid_use:
                    inner_cv = TimeSeriesSplit(n_splits=min(5, len(X_tr_fe)//2))
                    search = GridSearchCV(
                        estimator, grid_use, cv=inner_cv,
                        scoring='neg_root_mean_squared_error', n_jobs=-1
                    )
                    search.fit(X_tr_fe, y_tr_aligned)
                    model = search.best_estimator_
                    params = search.best_params_
                else:
                    model = estimator.fit(X_tr_fe, y_tr_aligned)
                    params = {}

                # metrics
                preds_tr = model.predict(X_tr_fe)
                preds_te = model.predict(X_te_fe)
                ins = compute_metrics(y_tr_aligned.values, preds_tr)
                oos = compute_metrics(y_te_aligned.values, preds_te)

                rec = {
                    'Mode': mode, 'Model': name,
                    'Lags': use_lags, 'Delta': use_delta,
                    'Window': window_type,
                    'TrainEnd': df.index[start],
                    'IN_n': len(y_tr_aligned), 'OOS_n': len(y_te_aligned),
                    **{f"IN_{k}": v for k, v in ins.items()},
                    **{f"OOS_{k}": v for k, v in oos.items()},
                    **params
                }
                records.append(rec)
    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# GLOBAL SELECTION + FINAL FIT + SUMMARY
# ---------------------------------------------------------------------------
def select_and_finalize(
    df: pd.DataFrame,
    target_col: str,
    metrics_df: pd.DataFrame
):
    # aggregate SSE over OOS
    md = metrics_df.copy()
    md['OOS_SSE'] = md['OOS_RMSE']**2 * md['OOS_n']
    agg = md.groupby(['Mode','Model','Lags','Delta','Window']).agg(
        OOS_SSE=('OOS_SSE','sum'),
        OOS_n=('OOS_n','sum')
    )
    agg['Agg_RMSE'] = np.sqrt(agg['OOS_SSE'] / agg['OOS_n'])
    best_cfg = agg['Agg_RMSE'].idxmin()
    logging.info("Best config by aggregated OOS RMSE: %s -> %.6f", best_cfg, agg.loc[best_cfg,'Agg_RMSE'])

    # final training on full dataset
    df_full = df.copy()
    X_full = df_full.drop(columns=[target_col])
    y_full = df_full[target_col]

    pipe = build_feature_pipeline(best_cfg[0], best_cfg[2], best_cfg[3], list(X_full.columns))
    X_fe = pd.DataFrame(pipe.fit_transform(X_full), index=X_full.index).dropna()
    y_al = y_full.reindex(X_fe.index)

    est, grid = LINEAR_MODELS[best_cfg[1]]
    if grid:
        cv = TimeSeriesSplit(n_splits=min(5, len(X_fe)//2))
        search = GridSearchCV(
            est, grid, cv=cv,
            scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        search.fit(X_fe, y_al)
        final_model = search.best_estimator_
        logging.info('Final model %s params: %s', best_cfg[1], search.best_params_)
    else:
        final_model = est.fit(X_fe, y_al)
        logging.info('Final model %s without hyperparams', best_cfg[1])

    # save model details
    params = final_model.get_params()
    coefs = getattr(final_model, 'coef_', None)
    intercept = getattr(final_model, 'intercept_', None)
    with open('final_model_info.txt','w') as f:
        f.write(f"Params: {params}\nCoefficients: {coefs}\nIntercept: {intercept}\n")
    logging.info("Saved final model info to final_model_info.txt")

    # statsmodels summary for inference
    X_sm = sm.add_constant(X_fe)
    smm = sm.OLS(y_al.values, X_sm).fit()
    with open('model_summary.txt','w') as f:
        f.write(str(smm.summary()))
    logging.info("Saved model summary to model_summary.txt")

    # plot actual vs predicted at t+1
    preds = final_model.predict(X_fe)
    preds_series = pd.Series(preds, index=X_fe.index)
    plt.figure(figsize=(10,6))
    plt.plot(y_al.index, y_al, label='Actual (t+1)')
    plt.plot(preds_series.index, preds_series, label='Predicted (t+1)')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.tight_layout()
    plt.savefig('actual_vs_pred.png')
    logging.info("Saved prediction plot to actual_vs_pred.png")

# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='State-of-the-art linear forecasting with walk-forward evaluation'
    )
    parser.add_argument('--x', required=True, help='Path to features file')
    parser.add_argument('--y', required=True, help='Path to target file')
    parser.add_argument('--target', default='Y', help='Target column name')
    parser.add_argument('--window', choices=['expanding','rolling'], default='expanding',
                        help='Window type for walk-forward')
    parser.add_argument('--train_periods', type=int, default=36,
                        help='Initial training window length in observations')
    parser.add_argument('--test_periods', type=int, default=12,
                        help='Test window length in observations')
    args = parser.parse_args()

    loader = DataLoader(Path(args.x), Path(args.y))
    df = loader.load_panel(target_col=args.target, shift_target=True)

    metrics_df = evaluate_all_combinations(
        df, args.target,
        modes=['raw','static_z','rolling_z'],
        train_periods=args.train_periods,
        test_periods=args.test_periods,
        window_type=args.window
    )
    metrics_df.to_csv('all_metrics.csv', index=False)
    logging.info("All metrics saved to all_metrics.csv")

    select_and_finalize(df, args.target, metrics_df)

if __name__ == '__main__':
    main()
