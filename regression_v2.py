# tp_full_stack_refined_v2.py  — Ultimate state-of-the-art term-premium forecaster
# =====================================================================
# USAGE:
#   python tp_full_stack_refined_v2.py --x X.parquet --y Y.parquet --target term_prem \
#       --cv-type expanding --splits 5 --gap 1 --optuna-trials 50
# =====================================================================

from __future__ import annotations
import argparse, logging, os, warnings, joblib, json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import shap
import optuna
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, LinearRegression
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    BaseCrossValidator, GridSearchCV, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from lightgbm import LGBMRegressor

# ---------------------------------------------------------------------------
# CONFIG & LOGGER
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# DATA LOADER
# ---------------------------------------------------------------------------
class DataLoader:
    def __init__(self, x_path: Path, y_path: Path):
        self.x_df = self._read(x_path)
        self.y_df = self._read(y_path)
        df = self.x_df.join(self.y_df, how="inner").sort_index()
        self.full = df.ffill().dropna()
        logging.info("Data loaded: %d observations × %d features", *self.full.shape)

    @staticmethod
    def _read(p: Path) -> pd.DataFrame:
        if p.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(p)
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p, index_col=0, parse_dates=True)
        raise ValueError(f"Unsupported file type {p.suffix}")

    def split(self, test_years: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cut = -12 * test_years
        return self.full.iloc[:cut], self.full.iloc[cut:]

# ---------------------------------------------------------------------------
# FEATURE TRANSFORMS
# ---------------------------------------------------------------------------
class Lagger(TransformerMixin, BaseEstimator):
    def __init__(self, lags: int = 2): self.lags = lags
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = pd.concat(
            {f"{c}_lag{l}": X[c].shift(l)
             for c in X.columns for l in range(1, self.lags+1)},
            axis=1
        )
        return df

class RollingZ(TransformerMixin, BaseEstimator):
    def __init__(self, window: int = 36): self.w = window
    def fit(self, X, y=None): return self
    def transform(self, X):
        mu = X.rolling(self.w, min_periods=self.w//2).mean()
        sd = X.rolling(self.w, min_periods=self.w//2).std()
        return (X - mu) / sd

# build feature pipeline inside model pipelines

def feature_pipe(lags: int = 2, rz_win: int = 36) -> ColumnTransformer:
    return ColumnTransformer([
        ("lag", Lagger(lags=lags), slice(0, None)),
        ("rz", RollingZ(window=rz_win), slice(0, None)),
    ], remainder="drop")

# ---------------------------------------------------------------------------
# CUSTOM CV
# ---------------------------------------------------------------------------
class PurgedWalkForwardCV(BaseCrossValidator):
    def __init__(self, splits: int = 5, gap: int = 1):
        self.splits, self.gap = splits, gap
    def split(self, X, y=None, groups=None):
        n = len(X)
        size = n // (self.splits + 1)
        for i in range(self.splits):
            tr_end = size * (i+1) - self.gap
            tr_idx = np.arange(0, tr_end)
            te_start = tr_end + self.gap
            te_end = min(te_start + size, n)
            yield tr_idx, np.arange(te_start, te_end)
    def get_n_splits(self, X=None, y=None, groups=None): return self.splits

class ExpandingPurgedCV(PurgedWalkForwardCV):
    def split(self, X, y=None, groups=None):
        n = len(X)
        size = n // (self.splits + 1)
        for i in range(self.splits):
            tr_end = size * (i+1) - self.gap
            tr_idx = np.arange(0, tr_end)
            te_start = tr_end + self.gap
            te_end = min(te_start + size, n)
            yield tr_idx, np.arange(te_start, te_end)

# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
def reg_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        'RMSE': mean_squared_error(y_true, y_pred, squared=False),
        'MAE':  mean_absolute_error(y_true, y_pred),
        'R2':   r2_score(y_true, y_pred)
    }

# ---------------------------------------------------------------------------
# OPTUNA FOR LIGHTGBM
# ---------------------------------------------------------------------------
def tune_lgb_optuna(
    X: pd.DataFrame, y: pd.Series, cv: BaseCrossValidator, trials: int
) -> Dict[str, Any]:
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 400]),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 16, 64),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'random_state': SEED,
            'n_jobs': -1
        }
        pipe = Pipeline([
            ('feat', feature_pipe()),
            ('sc', StandardScaler(with_mean=False)),
            ('lgb', LGBMRegressor(**params))
        ])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='r2', n_jobs=-1)
        return np.mean(scores)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials)
    logging.info("LGB Optuna best R2: %.4f", study.best_value)
    return study.best_params

# ---------------------------------------------------------------------------
# SHAP DUMP
# ---------------------------------------------------------------------------
def shap_dump(model, X, outdir: str = "explain", nsample: int = 200):
    os.makedirs(outdir, exist_ok=True)
    expl = shap.Explainer(model, X.iloc[:nsample])
    sv = expl(X.iloc[:nsample])
    shap.summary_plot(sv, X.iloc[:nsample], show=False)
    shap.save_html(Path(outdir)/"shap.html", sv)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run(cfg):
    t0 = datetime.now()
    # load & split
    dl = DataLoader(cfg.x, cfg.y)
    train, test = dl.split(cfg.holdout)
    ycol = cfg.target
    Xtr_raw, ytr = train.drop(columns=[ycol]), train[ycol]
    Xte_raw, yte = test.drop(columns=[ycol]), test[ycol]

    # choose CV
    CV = ExpandingPurgedCV if cfg.cv_type=='expanding' else PurgedWalkForwardCV
    cv = CV(splits=cfg.splits, gap=cfg.gap)
    logging.info("CV: %s splits=%d gap=%d", cfg.cv_type, cfg.splits, cfg.gap)

    # dummy baseline
    dummy = DummyRegressor(strategy='mean')
    d_scores = cross_val_score(dummy, Xtr_raw, ytr, cv=cv, scoring='r2', n_jobs=-1)
    logging.info("Dummy mean R2: %.4f", np.mean(d_scores))

    # GRID-SEARCHED LINEAR PIPELINES
    models: Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]] = {}

    # Ridge pipe with feature tuning
    ridge_pipe = Pipeline([
        ('feat', feature_pipe()),
        ('sc', StandardScaler(with_mean=False)),
        ('mdl', Ridge(random_state=SEED))
    ])
    ridge_params = {
        'feat__lag__lags': [1,2,3],
        'feat__rz__window': [24,36],
        'mdl__alpha': [0.01,0.1,1.0]
    }
    models['Ridge'] = (ridge_pipe, ridge_params)

    # Lasso pipe
    lasso_pipe = Pipeline([
        ('feat', feature_pipe()),
        ('sc', StandardScaler(with_mean=False)),
        ('mdl', Lasso(random_state=SEED))
    ])
    lasso_params = {
        'feat__lag__lags': [1,2,3],
        'feat__rz__window': [24,36],
        'mdl__alpha': [0.001,0.01,0.1]
    }
    models['Lasso'] = (lasso_pipe, lasso_params)

    # ElasticNet pipe
    enet_pipe = Pipeline([
        ('feat', feature_pipe()),
        ('sc', StandardScaler(with_mean=False)),
        ('mdl', ElasticNet(random_state=SEED))
    ])
    enet_params = {
        'feat__lag__lags': [1,2],
        'feat__rz__window': [36],
        'mdl__alpha': [0.01,0.1],
        'mdl__l1_ratio': [0.2,0.5]
    }
    models['ElasticNet'] = (enet_pipe, enet_params)

    # PCR: PCA + Ridge
    pcr_pipe = Pipeline([
        ('feat', feature_pipe()),
        ('sc', StandardScaler(with_mean=False)),
        ('pca', PCA()),
        ('mdl', Ridge(random_state=SEED))
    ])
    pcr_params = {
        'feat__lag__lags': [2],
        'feat__rz__window': [36],
        'pca__n_components': [3,5,8],
        'mdl__alpha': [0.1,1.0]
    }
    models['PCR'] = (pcr_pipe, pcr_params)

    # OLS baseline
    ols_pipe = Pipeline([
        ('feat', feature_pipe()),
        ('sc', StandardScaler(with_mean=False)),
        ('mdl', LinearRegression())
    ])
    models['OLS'] = (ols_pipe, {})

    # tune linear models
    results = []
    fitted = {}
    for name, (pipe, params) in models.items():
        if params:
            gs = GridSearchCV(pipe, params, cv=cv, scoring='r2', n_jobs=-1)
            gs.fit(Xtr_raw, ytr)
            best = gs.best_estimator_
            logging.info("%s best params: %s", name, gs.best_params_)
        else:
            best = pipe.fit(Xtr_raw, ytr)
        fitted[name] = best
        y_pred = best.predict(Xte_raw)
        results.append({'Model': name, **reg_metrics(yte, y_pred)})

    # LightGBM via Optuna
    if cfg.optuna_trials > 0:
        lgb_params = tune_lgb_optuna(Xtr_raw, ytr, cv, cfg.optuna_trials)
        lgb_pipe = Pipeline([
            ('feat', feature_pipe()),
            ('sc', StandardScaler(with_mean=False)),
            ('lgb', LGBMRegressor(**lgb_params))
        ])
        lgb_pipe.fit(Xtr_raw, ytr)
        fitted['LGB'] = lgb_pipe
        y_pred = lgb_pipe.predict(Xte_raw)
        results.append({'Model': 'LGB', **reg_metrics(yte, y_pred)})

    # STACKING the top performers
    top_models = [(n, fitted[n]) for n in ['Ridge','Lasso','ElasticNet'] if n in fitted]
    stack_pipe = Pipeline([
        ('feat', feature_pipe()),
        ('sc', StandardScaler(with_mean=False)),
        ('stack', StackingRegressor(
            estimators=top_models,
            final_estimator=Ridge(alpha=0.1, random_state=SEED),
            passthrough=True
        ))
    ])
    stack_pipe.fit(Xtr_raw, ytr)
    y_pred = stack_pipe.predict(Xte_raw)
    results.append({'Model': 'Stack', **reg_metrics(yte, y_pred)})
    fitted['Stack'] = stack_pipe

    # leaderboard
    df_lb = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)
    df_lb.to_csv('leaderboard_v2.csv', index=False)
    logging.info("\n%s", df_lb.to_string(index=False))

    # save final model
    best_name = df_lb.loc[0,'Model']
    joblib.dump({'model': fitted[best_name], 'cfg': vars(cfg)}, 'tp_model_best.joblib')

    # SHAP for best
    try:
        shap_dump(fitted[best_name], Xtr_raw)
        logging.info("SHAP saved for %s", best_name)
    except Exception as e:
        logging.warning("SHAP error: %s", e)

    logging.info("All done in %s", datetime.now() - t0)

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="State-of-the-art term-premium forecasting stack"
    )
    p.add_argument("--x", type=Path, required=True)
    p.add_argument("--y", type=Path, required=True)
    p.add_argument("--target", type=str, default="term_prem")
    p.add_argument("--holdout", type=int, default=2,
                   help="years for OOS test")
    p.add_argument("--cv-type", choices=['purged','expanding'], default='expanding',
                   help="CV strategy")
    p.add_argument("--splits", type=int, default=5,
                   help="number of CV splits")
    p.add_argument("--gap", type=int, default=1,
                   help="months purge between train/test")
    p.add_argument("--optuna-trials", type=int, default=0,
                   help="run Optuna LGB tuning (0 to disable)")
    cfg = p.parse_args()
    run(cfg)
