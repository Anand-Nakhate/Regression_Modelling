from __future__ import annotations
import argparse, logging, os, warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Logging & warnings
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
    handlers=[logging.StreamHandler()]
)

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# DATA LOADING & PREP
# ---------------------------------------------------------------------------
class DataLoader:
    def __init__(self, x_path: Path, y_path: Path):
        self.x_df = self._read(x_path)
        self.y_df = self._read(y_path)
        self.panel = self._align(self.x_df, self.y_df)

    @staticmethod
    def _read(p: Path) -> pd.DataFrame:
        if p.suffix.lower() in {'.parquet', '.pq'}:
            return pd.read_parquet(p)
        if p.suffix.lower() == '.csv':
            return pd.read_csv(p, index_col=0, parse_dates=True)
        raise ValueError(f"Unsupported file type {p.suffix}")

    @staticmethod
    def _align(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        df = x.join(y, how='inner').sort_index()
        df = df.ffill().dropna()
        logging.info("Aligned data: %d obs × %d features", *df.shape)
        return df

# ---------------------------------------------------------------------------
# FEATURE ENGINEERING MODES
# ---------------------------------------------------------------------------
class StaticZ(TransformerMixin, BaseEstimator):
    def __init__(self, with_mean: bool=True, with_std: bool=True):
        self.with_mean, self.with_std = with_mean, with_std
    def fit(self, X, y=None): return self
    def transform(self, X):
        mu = X.mean() if self.with_mean else 0
        sd = X.std() if self.with_std else 1
        return (X - mu) / sd

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
        df = pd.concat(
            {f"{col}_lag{lag}": X[col].shift(lag)
             for col in X.columns
             for lag in range(1, self.lags+1)}, axis=1
        )
        return df

class Delta(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None): return self
    def transform(self, X): return X.diff()

def build_feature_pipeline(mode: str, use_lags: bool, use_delta: bool) -> Pipeline:
    transformers = []
    # Base transformation
    if mode == 'raw':
        transformers.append(( 'identity', 'passthrough', slice(0, 1) ))
    elif mode == 'static_z':
        transformers.append(( 'static_z', StaticZ(), slice(None) ))
    elif mode == 'rolling_z':
        transformers.append(( 'rolling_z', RollingZ(window=36), slice(None) ))
    # Optionally add lags/delta
    if use_lags:
        transformers.append(( 'lag', Lagger(lags=3), slice(None) ))
    if use_delta:
        transformers.append(( 'delta', Delta(), slice(None) ))
    # Combine
    ct = ColumnTransformer(transformers, remainder='drop')
    # Scale
    pipe = Pipeline([('features', ct), ('scale', StandardScaler(with_mean=False))])
    return pipe

# ---------------------------------------------------------------------------
# MODEL DEFINITION & PARAM GRIDS
# ---------------------------------------------------------------------------
LINEAR_MODELS = {
    'OLS': (LinearRegression(), {}),
    'Ridge': (Ridge(random_state=SEED), {'alpha': [0.01, 0.1, 1.0, 10.0]}),
    'Lasso': (Lasso(random_state=SEED), {'alpha': [0.001, 0.01, 0.1, 1.0]}),
    'ElasticNet': (ElasticNet(random_state=SEED), {
        'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]
    }),
    'BayesianRidge': (BayesianRidge(), {}),
    'PCR': (
        Pipeline([('sc', StandardScaler()), ('pca', PCA()), ('lr', LinearRegression())]),
        {'pca__n_components': [2, 5, 10, 0.95]}
    ),
    'PLS': (Pipeline([('pls', PCA()), ('lr', LinearRegression())]),
            {'pls__n_components': [2, 5, 10]})
}

# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

# ---------------------------------------------------------------------------
# ROLLING & EXPANDING EVALUATION
# ---------------------------------------------------------------------------

def walk_forward_evaluate(
    data: pd.DataFrame,
    target_col: str,
    mode: str,
    use_lags: bool,
    use_delta: bool,
    window_type: str,
    train_size: int,
    test_size: int
) -> Tuple[pd.DataFrame, dict]:
    """
    Perform walk-forward evaluation on time-series data.
    window_type: 'expanding' or 'rolling'
    train_size: number of periods for initial train
    test_size: number of periods to forecast each step
    Returns a DataFrame of metrics per step and overall best model info.
    """
    metrics_list = []
    param_records = []
    n = len(data)
    start = train_size
    idx = list(range(start, n - test_size + 1, test_size))
    for end_train in idx:
        # select train/test
        if window_type == 'expanding':
            train_idx = list(range(0, end_train))
        else:  # rolling
            train_idx = list(range(end_train - train_size, end_train))
        test_idx = list(range(end_train, end_train + test_size))

        df_train = data.iloc[train_idx]
        df_test = data.iloc[test_idx]
        X_train = df_train.drop(columns=[target_col])
        y_train = df_train[target_col]
        X_test = df_test.drop(columns=[target_col])
        y_test = df_test[target_col]

        # build pipeline + model search
        feat_pipe = build_feature_pipeline(mode, use_lags, use_delta)
        X_tr_fe = feat_pipe.fit_transform(X_train)
        X_te_fe = feat_pipe.transform(X_test)

        for name, (estimator, grid) in LINEAR_MODELS.items():
            # grid search if grid not empty
            if grid:
                tscv = TimeSeriesSplit(n_splits=3)
                gs = GridSearchCV(
                    estimator, grid, cv=tscv, scoring='neg_root_mean_squared_error'
                )
                gs.fit(X_tr_fe, y_train)
                best_model = gs.best_estimator_
                best_params = gs.best_params_
            else:
                estimator.random_state = SEED if hasattr(estimator, 'random_state') else None
                best_model = estimator.fit(X_tr_fe, y_train)
                best_params = {}

            preds = best_model.predict(X_te_fe)
            m = reg_metrics(y_test.values, preds)
            m.update({'Mode': mode, 'Model': name,
                      'Lags': use_lags, 'Delta': use_delta,
                      'WindowType': window_type,
                      'TrainEnd': data.index[end_train],
                      **best_params})
            metrics_list.append(m)

    metrics_df = pd.DataFrame(metrics_list)
    # aggregate average metrics
    avg_metrics = metrics_df.groupby(['Mode', 'Model', 'Lags', 'Delta', 'WindowType']).mean()
    # pick best configuration by RMSE
    best_cfg = avg_metrics['RMSE'].idxmin()
    best_info = avg_metrics.loc[best_cfg].to_dict()
    return metrics_df, {'BestConfig': best_cfg, 'BestMetrics': best_info}

# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_actual_vs_pred(data: pd.DataFrame, target_col: str, predictions: pd.Series, out_path: Path):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data[target_col], label='Actual')
    plt.plot(predictions.index, predictions, label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(out_path)
    logging.info("Saved plot to %s", out_path)

# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------

def main(args):
    dl = DataLoader(Path(args.x), Path(args.y))
    df = dl.panel

    # parameters
    train_size = args.train_size * 12  # years to months
    test_size = args.test_size
    window_type = args.window_type  # 'expanding' or 'rolling'

    results, best = walk_forward_evaluate(
        df, args.target, args.mode,
        args.use_lags, args.use_delta,
        window_type, train_size, test_size
    )

    # output results
    results.to_csv('rolling_evaluation.csv', index=False)
    logging.info('Evaluation complete. Best config: %s', best)

    # final model training on full dataset with best config
    mode, model_name, lags, delta, wtype = best['BestConfig']
    feat_pipe = build_feature_pipeline(mode, lags, delta)
    X_full = feat_pipe.fit_transform(df.drop(columns=[args.target]))
    y_full = df[args.target]
    estimator, grid = LINEAR_MODELS[model_name]
    if grid:
        gs = GridSearchCV(estimator, grid, cv=TimeSeriesSplit(n_splits=3),
                          scoring='neg_root_mean_squared_error')
        gs.fit(X_full, y_full)
        final_model = gs.best_estimator_
        logging.info('Trained final model %s with params %s', model_name, gs.best_params_)
    else:
        final_model = estimator.fit(X_full, y_full)

    # predictions & plot
    preds = pd.Series(final_model.predict(X_full), index=df.index)
    plot_actual_vs_pred(df, args.target, preds, Path('actual_vs_pred.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rolling/Expanding linear model evaluation')
    parser.add_argument('--x', required=True, help='Path to X features file')
    parser.add_argument('--y', required=True, help='Path to Y target file')
    parser.add_argument('--target', default='Y', help='Target column name')
    parser.add_argument('--mode', choices=['raw','static_z','rolling_z'], default='rolling_z')
    parser.add_argument('--use_lags', action='store_true', help='Include lag features')
    parser.add_argument('--use_delta', action='store_true', help='Include delta features')
    parser.add_argument('--window_type', choices=['expanding','rolling'], default='expanding')
    parser.add_argument('--train_size', type=int, default=3, help='Training window size in years')
    parser.add_argument('--test_size', type=int, default=12, help='Test window size in periods')
    args = parser.parse_args()
    main(args)
