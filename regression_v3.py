#!/usr/bin/env python3
# tp_model.py

from __future__ import annotations
import argparse, logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import (LinearRegression, Ridge, Lasso,
                                  ElasticNet, BayesianRidge)
from sklearn.model_selection import GridSearchCV, BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import make_scorer, mean_squared_error

# ─────────────────────────────────────────
# logging & seed
# ─────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
    handlers=[logging.StreamHandler()]
)

# ─────────────────────────────────────────
# Data loading & splitting
# ─────────────────────────────────────────
class DataLoader:
    def __init__(self, x_path: Path, y_path: Path):
        self.x = self._read(x_path)
        self.y = self._read(y_path)
        self.data = (
            self.x.join(self.y, how="inner")
                  .sort_index()
                  .ffill()
                  .dropna()
        )
        logging.info("Loaded and aligned: %d obs × %d features",
                     len(self.data), self.x.shape[1])

    @staticmethod
    def _read(p: Path) -> pd.DataFrame:
        if p.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(p)
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p, index_col=0, parse_dates=True)
        raise ValueError(f"Unsupported file type: {p.suffix}")

    def split(self, holdout_years: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cut = -12 * holdout_years
        return self.data.iloc[:cut], self.data.iloc[cut:]

# ─────────────────────────────────────────
# Feature transformers
# ─────────────────────────────────────────
class Lagger(TransformerMixin, BaseEstimator):
    def __init__(self, lags: int = 3):
        self.lags = lags
    def fit(self, X, y=None): return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = X.columns
        lagged = {
            f"{c}_lag{l}": X[c].shift(l)
            for c in cols for l in range(1, self.lags+1)
        }
        return pd.DataFrame(lagged, index=X.index)

class RollingZ(TransformerMixin, BaseEstimator):
    def __init__(self, window: int = 36):
        self.window = window
    def fit(self, X, y=None): return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        mu = X.rolling(self.window, self.window//2).mean()
        sd = X.rolling(self.window, self.window//2).std(ddof=0)
        return (X - mu) / sd

class Delta(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None): return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.diff()

def make_feature_pipeline(feature_cols: List[str]) -> Pipeline:
    """
    Builds a pipeline:
      1) parallel lag / rolling-z / delta on raw features
      2) drop NaNs
      3) center & scale all resulting columns
    """
    ct = ColumnTransformer([
        ("lag",   Lagger(lags=3),         feature_cols),
        ("rz",    RollingZ(window=36),     feature_cols),
        ("d1",    Delta(),                feature_cols),
    ], remainder="drop", n_jobs=-1)

    return Pipeline([
        ("feats", ct),
        ("scaler", StandardScaler())
    ])

# ─────────────────────────────────────────
# Purged walk-forward CV
# ─────────────────────────────────────────
class PurgedWalkForwardCV(BaseCrossValidator):
    def __init__(self, n_splits: int = 8, gap: int = 1):
        self.n_splits = n_splits
        self.gap = gap

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end = fold_size * (i + 1) - self.gap
            test_start = train_end + self.gap
            test_end = min(test_start + fold_size, n)
            yield (
                np.arange(train_end),
                np.arange(test_start, test_end)
            )

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

# ─────────────────────────────────────────
# Metrics & scorer
# ─────────────────────────────────────────
def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred, squared=False)

neg_rmse_scorer = make_scorer(rmse, greater_is_better=False)

# ─────────────────────────────────────────
# Model definitions & hyper-parameter grids
# ─────────────────────────────────────────
def gaussian_process() -> GaussianProcessRegressor:
    kernel = kernels.ConstantKernel(1.0, (1e-2, 1e3)) * kernels.RBF(1.0)
    return GaussianProcessRegressor(kernel=kernel,
                                    n_restarts_optimizer=3,
                                    random_state=SEED)

MODEL_BUILDERS: Dict[str, Tuple[BaseEstimator, Dict[str, List[Any]]]] = {
    "OLS":   (LinearRegression(), {}),
    "Ridge": (Ridge(random_state=SEED), {"model__alpha": [0.1, 1.0, 10.0]}),
    "Lasso": (Lasso(random_state=SEED), {"model__alpha": [0.001, 0.01, 0.1, 1.0]}),
    "Elastic": (
        ElasticNet(random_state=SEED),
        {
            "model__alpha":    [0.001, 0.01, 0.1, 1.0],
            "model__l1_ratio": [0.2, 0.5, 0.8]
        }
    ),
    "PCR": (
        Pipeline([
            ("pca", PCA()),
            ("lr",  LinearRegression())
        ]),
        {"pca__n_components": [0.90, 0.95, 0.99]}
    ),
    "PLS": (
        PCA(),  # placeholder – we’ll wrap below
        {}      # we'll handle PLS separately
    ),
    "SVR":   (SVR(), {"model__C": [0.1, 1.0, 10.0], "model__epsilon": [0.01, 0.1, 1.0]}),
    "GP":    (gaussian_process(), {}),
    "RF":    (
        RandomForestRegressor(random_state=SEED, n_jobs=-1),
        {"model__n_estimators": [200, 400, 800],
         "model__min_samples_leaf": [1, 2, 4]}
    ),
    "LGB":   (
        LGBMRegressor(objective="regression", random_state=SEED),
        {"model__learning_rate": [0.01, 0.04, 0.1],
         "model__n_estimators":  [200, 600, 1000]}
    ),
    "XGB":   (
        XGBRegressor(objective="reg:squarederror", random_state=SEED, n_jobs=-1),
        {"model__learning_rate": [0.01, 0.05, 0.1],
         "model__n_estimators":  [200, 600, 1000]}
    ),
    "Cat":   (
        CatBoostRegressor(loss_function="RMSE", random_state=SEED, verbose=False),
        {"model__learning_rate": [0.01, 0.05],
         "model__depth":         [4, 6, 8]}
    ),
    "NN":    (
        MLPRegressor(max_iter=1000, random_state=SEED),
        {"model__hidden_layer_sizes": [(32,), (64,32), (128,64,32)],
         "model__alpha":            [1e-4, 1e-3, 1e-2]}
    ),
}

# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def run(args: argparse.Namespace):
    start = datetime.now()
    # 1) load & split
    loader = DataLoader(args.x, args.y)
    train, test = loader.split(args.holdout)
    ycol = args.target

    Xtr_raw, ytr = train.drop(columns=[ycol]), train[ycol]
    Xte_raw, yte = test .drop(columns=[ycol]), test [ycol]

    # 2) build feature pipe
    fe_pipe = make_feature_pipeline(Xtr_raw.columns.tolist())

    # 3) set up CV
    tscv = PurgedWalkForwardCV(n_splits=8, gap=1)
    results: List[Dict[str, Any]] = []

    # 4) grid-search each model
    for name, (base_model, param_grid) in MODEL_BUILDERS.items():
        logging.info("→ Tuning %s", name)

        # special handling for PCR and PLS
        if name == "PCR":
            mdl = Pipeline([("fe", fe_pipe),
                            ("pca_lr", base_model)])
        elif name == "PLS":
            from sklearn.cross_decomposition import PLSRegression
            mdl = Pipeline([("fe", fe_pipe),
                            ("model", PLSRegression())])
            param_grid = {"model__n_components": [1, 2, 3, 4]}
        else:
            mdl = Pipeline([("fe", fe_pipe),
                            ("model", base_model)])

        if param_grid:
            search = GridSearchCV(
                mdl, param_grid,
                cv=tscv,
                scoring=neg_rmse_scorer,
                n_jobs=-1,
                verbose=0
            )
            search.fit(Xtr_raw, ytr)
            best_rmse = -search.best_score_
            best_est = search.best_estimator_
            best_params = search.best_params_
        else:
            # no grid → just fit once
            mdl.fit(Xtr_raw, ytr)
            preds = mdl.predict(Xtr_raw)
            best_rmse = rmse(ytr, preds)
            best_est = mdl
            best_params = {}

        results.append({
            "Model": name,
            "CV_RMSE": best_rmse,
            "BestParams": best_params,
            "Estimator": best_est
        })
        logging.info("   %s → CV RMSE = %.4f", name, best_rmse)

    # 5) pick best
    df = pd.DataFrame(results).sort_values("CV_RMSE")
    logging.info("\nLeaderboard:\n%s", df[["Model", "CV_RMSE", "BestParams"]].to_string(index=False))

    best_row = df.iloc[0]
    best_name = best_row.Model
    best_pipeline = best_row.Estimator
    logging.info("Selected best model: %s", best_name)

    # 6) static in-sample model
    static_pipe = Pipeline([("fe", fe_pipe),
                            ("model", LinearRegression())])
    static_pipe.fit(Xtr_raw, ytr)
    static_pred = static_pipe.predict(Xtr_raw)
    static_in_rmse = rmse(ytr, static_pred)
    logging.info("Static in-sample RMSE (no CV): %.4f", static_in_rmse)

    # 7) hold-out test
    yhat = best_pipeline.predict(Xte_raw)
    test_rmse = rmse(yte, yhat)
    logging.info("Hold-out (%s) RMSE = %.4f", best_name, test_rmse)

    # 8) save artifacts & report
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(df[["Model", "CV_RMSE"]]).to_csv(out_dir/"leaderboard.csv", index=False)
    best_pipeline.named_steps["model"].feature_names_in_ if hasattr(best_pipeline.named_steps["model"], "feature_names_in_") else None
    # you can dump the full pipeline via joblib if desired:
    # import joblib
    # joblib.dump(best_pipeline, out_dir/"best_pipeline.joblib")

    report = {
        "best_model": best_name,
        "cv_rmse": df.iloc[0].CV_RMSE,
        "static_in_rmse": static_in_rmse,
        "holdout_rmse": test_rmse,
        "train_size": len(Xtr_raw),
        "test_size": len(Xte_raw),
        "duration": str(datetime.now() - start)
    }
    with open(out_dir/"report.json", "w") as f:
        json.dump(report, f, indent=2)

    logging.info("Finished in %s", datetime.now() - start)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="State-of-the-art term-premium modeller with grid search"
    )
    p.add_argument("--x",      type=Path, required=True, help="Path to X (features)")
    p.add_argument("--y",      type=Path, required=True, help="Path to y (target)")
    p.add_argument("--target", type=str, default="Y", help="Name of target column")
    p.add_argument("--holdout",type=int, default=2, help="Years for OOS hold-out")
    args = p.parse_args()
    run(args)