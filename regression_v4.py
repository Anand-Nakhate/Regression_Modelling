# run.py  — State‑of‑the‑art term‑premium modeller
from __future__ import annotations
import os, json, joblib, hashlib, warnings, logging, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import optuna, structlog, shap

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import RidgeCV, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

SEED = 42
np.random.seed(SEED)
warnings.filterwarnings("ignore")

###############################################################################
# Logging & utilities
###############################################################################
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    processors=[structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                structlog.processors.JSONRenderer()]
)
log = structlog.get_logger()

def hash_run() -> str:
    return hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8]

###############################################################################
# Feature engineering blocks
###############################################################################
class Lagger(BaseEstimator, TransformerMixin):
    def __init__(self, lags: int = 3): self.lags = lags
    def fit(self, *_): return self
    def transform(self, X):
        out = {f"{c}_lag{l}": X[c].shift(l) for c in X.columns for l in range(1, self.lags+1)}
        return pd.DataFrame(out, index=X.index)

class RollingZ(BaseEstimator, TransformerMixin):
    def __init__(self, window: int = 36): self.w = window
    def fit(self, *_): return self
    def transform(self, X):
        mu = X.rolling(self.w, self.w//2).mean()
        sd = X.rolling(self.w, self.w//2).std()
        return (X - mu)/sd

class Delta(BaseEstimator, TransformerMixin):
    def fit(self, *_): return self
    def transform(self, X): return X.diff()

def make_feature_pipe(cols_raw: list[str], cols_macro: list[str]|None) -> Pipeline:
    blocks = [
        ("lags",   Lagger(3),       cols_raw),
        ("rz",     RollingZ(36),    cols_raw),
        ("delta",  Delta(),         cols_raw)
    ]
    if cols_macro:
        # macro factors go in untouched – they are already stationary
        blocks.append(("macro", "passthrough", cols_macro))
    return Pipeline([
        ("ct", ColumnTransformer(blocks, remainder="drop", sparse_threshold=0.3)),
        ("scale", RobustScaler(quantile_range=(10, 90)))
    ])

###############################################################################
# CV & metrics
###############################################################################
def reg_metrics(y, p):
    rmse = mean_squared_error(y, p, squared=False)
    mae  = mean_absolute_error(y, p)
    mape = np.mean(np.abs((y-p)/y))*100
    r2   = r2_score(y, p)
    hit  = (np.sign(np.diff(y)) == np.sign(np.diff(p))).mean()
    return dict(RMSE=rmse, MAE=mae, MAPE=mape, R2=r2, HitRatio=hit)

###############################################################################
# Model zoo & Optuna search spaces
###############################################################################
def optuna_space(trial, name: str):
    if name=="Ridge":
        return dict(alpha=trial.suggest_loguniform("alpha", 1e-3, 1e3))
    if name=="SVR":
        return dict(C=trial.suggest_loguniform("C", 1e-2, 1e2),
                    epsilon=trial.suggest_float("epsilon", 0.01, 0.2))
    if name=="XGB":
        return dict(max_depth=trial.suggest_int("max_depth",2,6),
                    min_child_weight=trial.suggest_int("mcw",1,10),
                    subsample=trial.suggest_float("subsample",0.5,1.0),
                    colsample_bytree=trial.suggest_float("colsample",0.5,1.0))
    if name=="LGB":
        return dict(num_leaves=trial.suggest_int("leaves",16,64),
                    subsample=trial.suggest_float("subsample",0.5,1.0),
                    colsample_bytree=trial.suggest_float("colsample",0.5,1.0))
    if name=="Cat":
        return dict(depth=trial.suggest_int("depth",4,8),
                    l2_leaf_reg=trial.suggest_float("l2",1,10))
    return {}   # default for simpler models

def build_model(name: str, params: dict):
    if name=="Ridge":   return RidgeCV(alphas=[params["alpha"]])
    if name=="SVR":     return SVR(C=params["C"], epsilon=params["epsilon"], kernel="rbf")
    if name=="RF":      return RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                                    max_features="sqrt", random_state=SEED, n_jobs=-1)
    if name=="XGB":     return XGBRegressor(objective="reg:squarederror", random_state=SEED,
                                           n_estimators=600, learning_rate=0.05, **params)
    if name=="LGB":     return LGBMRegressor(objective="regression", random_state=SEED,
                                           n_estimators=600, learning_rate=0.04, **params)
    if name=="Cat":     return CatBoostRegressor(loss_function="RMSE", random_state=SEED,
                                               iterations=600, learning_rate=0.05, verbose=False, **params)
    if name=="GP":      return GaussianProcessRegressor(kernel=ConstantKernel(1.0)*RBF(),
                                                       random_state=SEED)
    raise ValueError(name)

###############################################################################
# Training routine
###############################################################################
def train_one(model_name: str, X, y, feature_pipe, n_splits=8):
    cv = TimeSeriesSplit(n_splits=n_splits, gap=1, test_size=len(X)//(n_splits+1))
    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    def objective(trial):
        params = optuna_space(trial, model_name)
        est = build_model(model_name, params)
        scores = []
        for train_idx, val_idx in cv.split(X):
            Xt = feature_pipe.fit_transform(X.iloc[train_idx])
            Xv = feature_pipe.transform(X.iloc[val_idx])
            est.fit(Xt, y.iloc[train_idx])
            p = est.predict(Xv)
            scores.append(mean_squared_error(y.iloc[val_idx], p, squared=False))
        return np.mean(scores)
    study.optimize(objective, n_trials=30, timeout=600, show_progress_bar=False)
    best_params = study.best_trial.user_attrs.get("params", study.best_params)
    final_est = build_model(model_name, best_params)
    log.info("best_params", model=model_name, params=best_params)
    final_est.fit(feature_pipe.fit_transform(X), y)
    return final_est, best_params, study

###############################################################################
# Main orchestration
###############################################################################
def main(args):
    run_id = hash_run()
    log.info("starting", run_id=run_id)

    # ------------------------------------------------------------------ Load
    X_raw = pd.read_parquet(args.x)       if args.x.suffix in {".parquet",".pq"} \
        else pd.read_csv(args.x, index_col=0, parse_dates=True)
    y_raw = pd.read_parquet(args.y)       if args.y.suffix in {".parquet",".pq"} \
        else pd.read_csv(args.y, index_col=0, parse_dates=True)
    full = X_raw.join(y_raw, how="inner").ffill().dropna()
    X, y = full.drop(columns=[args.target]), full[args.target]

    # Separate possible macro columns (optional, they must already be in X)
    macro_cols = [c for c in X.columns if c.startswith("MACRO_")]
    feat_pipe = make_feature_pipe(X.columns.difference(macro_cols).tolist(), macro_cols)

    # ------------------------------------------------------------------ Split
    test_years = args.holdout
    cut = -12*test_years
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]

    # ------------------------------------------------------------------ Model loop
    board, artefacts = [], {}
    for mdl in ["Ridge","SVR","RF","LGB","XGB","Cat"]:
        est, params, study = train_one(mdl, X_train, y_train, feat_pipe)
        preds = est.predict(feat_pipe.transform(X_test))
        board.append(dict(Model=mdl, **reg_metrics(y_test, preds)))
        artefacts[mdl] = dict(model=est, params=params, study=study.trials_dataframe())

    lb = pd.DataFrame(board).sort_values("RMSE")
    lb.to_csv(f"leaderboard_{run_id}.csv", index=False)
    log.info("leaderboard", table=lb.to_dict(orient="records"))

    # ------------------------------------------------------------------ Static / full-sample fit
    best_mdl = lb.iloc[0].Model
    final_est = artefacts[best_mdl]["model"]
    final_est.fit(feat_pipe.fit_transform(X), y)     # ***static training***
    joblib.dump({"model": final_est, "feature_pipe": feat_pipe},
                f"tp_model_{run_id}.joblib")

    # ------------------------------------------------------------------ Explainability
    try:
        explainer = shap.Explainer(final_est, feat_pipe.transform(X.iloc[:200]))
        sv = explainer(feat_pipe.transform(X.iloc[:200]))
        shap.save_html(f"shap_{run_id}.html", sv)
    except Exception as e:
        log.warning("shap_failed", error=str(e))

    # ------------------------------------------------------------------ Persist CV traces
    for mdl, art in artefacts.items():
        art["study"].to_parquet(f"cv_traces_{mdl}_{run_id}.parquet")

    log.info("done", run_id=run_id, best=best_mdl)

###############################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--x", type=Path, required=True)
    p.add_argument("--y", type=Path, required=True)
    p.add_argument("--target", type=str, default="TP")
    p.add_argument("--holdout", type=int, default=2)
    args = p.parse_args()
    main(args)
