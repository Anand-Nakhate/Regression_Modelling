from __future__ import annotations
import argparse, logging, os, warnings, joblib, json
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import optuna, shap
from tqdm import tqdm

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, PLSRegression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso,
                                  ElasticNet, BayesianRidge)
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor

import statsmodels.api as sm
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)8s | %(message)s",
                    handlers=[logging.StreamHandler()])

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# DATA ----------------------------------------------------------------------
# ---------------------------------------------------------------------------
class DataLoader:
    def __init__(self, x_path: Path, y_path: Path):
        self.x_df = self._read(x_path)
        self.y_df = self._read(y_path)
        self.full = (self.x_df.join(self.y_df, how="inner")  # align
                               .sort_index()
                               .ffill()                     # small gaps
                               .dropna())
        logging.info("Final aligned panel: %d obs × %d features",
                     len(self.full), self.x_df.shape[1])

    @staticmethod
    def _read(p: Path) -> pd.DataFrame:
        if p.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(p)
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p, index_col=0, parse_dates=True)
        raise ValueError(f"Unsupported file type {p.suffix}")

    def split(self, test_years: int = 2):
        cut = -12 * test_years
        return self.full.iloc[:cut], self.full.iloc[cut:]

# ---------------------------------------------------------------------------
# FEATURE ENGINEERING --------------------------------------------------------
# ---------------------------------------------------------------------------
class Lagger(TransformerMixin, BaseEstimator):
    def __init__(self, lags: int = 3):
        self.lags = lags
    def fit(self, *_): return self
    def transform(self, X):
        out = {f"{c}_lag{l}": X[c].shift(l) for c in X.columns for l in range(1, self.lags+1)}
        return pd.DataFrame(out, index=X.index)

class RollingZ(TransformerMixin, BaseEstimator):
    def __init__(self, window: int = 36):
        self.w = window
    def fit(self, *_): return self
    def transform(self, X):
        mu = X.rolling(self.w, self.w//2).mean()
        sd = X.rolling(self.w, self.w//2).std()
        return (X - mu)/sd

class Delta(TransformerMixin, BaseEstimator):
    def fit(self, *_): return self
    def transform(self, X): return X.diff()

def feature_pipe(raw_X: pd.DataFrame) -> Pipeline:
    return Pipeline([
        ("ct", ColumnTransformer([
            ("lag", Lagger(3), raw_X.columns),
            ("rz",  RollingZ(36), raw_X.columns),
            ("d1",  Delta(), raw_X.columns),
        ], remainder="drop")),
        ("sc", StandardScaler(with_mean=False))  # keeps sparse matrix slim
    ])

# ---------------------------------------------------------------------------
# TIME‑SERIES CV (purged walk‑forward) --------------------------------------
# ---------------------------------------------------------------------------
class PurgedWalkForwardCV(BaseCrossValidator):
    def __init__(self, splits: int = 8, gap: int = 1):
        self.splits, self.gap = splits, gap
    def split(self, X, *_):
        n, size = len(X), len(X)//(self.splits+1)
        for i in range(self.splits):
            tr_end = size*(i+1)-self.gap
            te_start, te_end = tr_end+self.gap, min(tr_end+self.gap+size, n)
            yield np.arange(tr_end), np.arange(te_start, te_end)
    def get_n_splits(self, *_): return self.splits

# ---------------------------------------------------------------------------
# METRICS -------------------------------------------------------------------
# ---------------------------------------------------------------------------
def reg_metrics(y, p):
    rmse = mean_squared_error(y, p, squared=False)
    mae  = mean_absolute_error(y, p)
    mape = (np.abs((y-p)/y).mean())*100
    r2   = r2_score(y, p)
    dir_acc = (np.sign(y.diff()) == np.sign(pd.Series(p, index=y.index).diff())).mean()
    return dict(RMSE=rmse, MAE=mae, MAPE=mape, R2=r2, HitRatio=dir_acc)

def port_metrics(pred, ret):
    sig = np.sign(pred - pred.mean())
    pnl = sig.shift(1)*ret
    ann = 12
    sharpe = np.sqrt(ann)*pnl.mean()/pnl.std(ddof=0)
    sortino= np.sqrt(ann)*pnl.mean()/pnl[pnl<0].std(ddof=0)
    mdd    = (pnl.cumsum().cummax()-pnl.cumsum()).max()
    return dict(Sharpe=sharpe, Sortino=sortino, MaxDD=mdd)

# ---------------------------------------------------------------------------
# MODEL ZOO -----------------------------------------------------------------
# ---------------------------------------------------------------------------
def gaussian_process():
    kernel = C(1.0, (1e-2, 1e3))*RBF(length_scale=1.0)
    return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=SEED)

MODELS: Dict[str, callable] = {
    "OLS"      : lambda _: LinearRegression(),
    "Ridge"    : lambda ps: Ridge(**ps),
    "Lasso"    : lambda ps: Lasso(**ps),
    "Elastic"  : lambda ps: ElasticNet(**ps),
    "BayesR"   : lambda _: BayesianRidge(),
    "PCR"      : lambda ps: Pipeline([("sc",StandardScaler()), ("pca",PCA(0.95)), ("lr",LinearRegression())]),
    "PLS"      : lambda ps: PLSRegression(n_components=ps.get("n_comp",2)),
    "SVR"      : lambda ps: Pipeline([("sc",StandardScaler()), ("svr",SVR(C=ps["C"], epsilon=0.05))]),
    "GP"       : lambda _: gaussian_process(),
    "RF"       : lambda ps: RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                                  random_state=SEED, n_jobs=-1),
    "LGB"      : lambda ps: LGBMRegressor(objective="regression", random_state=SEED,
                                         n_estimators=600, learning_rate=0.04,
                                         subsample=0.8, colsample_bytree=0.8, **ps),
    "XGB"      : lambda ps: XGBRegressor(objective="reg:squarederror", random_state=SEED,
                                         n_estimators=600, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8, **ps),
    "Cat"      : lambda ps: CatBoostRegressor(loss_function="RMSE", verbose=False,
                                              random_state=SEED, iterations=600,
                                              learning_rate=0.05, depth=6, **ps),
}

def stacked():
    base = [
        ("ridge", Ridge(alpha=1.0)),
        ("lgb", MODELS["LGB"]({})),
        ("svr", MODELS["SVR"]({"C":1.0}))
    ]
    return StackingRegressor(estimators=base, final_estimator=ElasticNet(alpha=0.01,l1_ratio=0.1),
                             n_jobs=-1, passthrough=True)

# VAR benchmark (needs X & y together)
class VARBench:
    def __init__(self, p=6): self.p=p
    def fit(self, X, y):
        df = X.copy(); df[y.name]=y
        self.m = sm.tsa.VAR(df).fit(maxlags=self.p, ic="aic")
        return self
    def predict(self, X): return self.m.forecast(X.values[-self.m.k_ar:], 1)[:, -1]

# ---------------------------------------------------------------------------
# TRAIN & EVALUATE ----------------------------------------------------------
# ---------------------------------------------------------------------------
def cv_score(model, X, y, cv):
    scores=[]
    for tr, te in cv.split(X):
        m=model
        m.fit(X.iloc[tr], y.iloc[tr])
        p=m.predict(X.iloc[te])
        scores.append(reg_metrics(y.iloc[te], p))
    return pd.DataFrame(scores).mean().to_dict()

def shap_dump(model, X, outdir="explain"):
    os.makedirs(outdir, exist_ok=True)
    explainer = shap.Explainer(model, X[:200])
    sv = explainer(X[:200])
    shap.summary_plot(sv, X[:200], show=False)
    shap.save_html(Path(outdir)/"shap.html", sv)

# ---------------------------------------------------------------------------
# MAIN ----------------------------------------------------------------------
# ---------------------------------------------------------------------------
def run(cfg):
    tic=datetime.now()
    dl = DataLoader(cfg.x, cfg.y)
    train, test = dl.split(cfg.holdout)
    ycol=cfg.target

    Xtr_raw, ytr = train.drop(columns=[ycol]), train[ycol]
    Xte_raw, yte = test.drop(columns=[ycol]), test[ycol]

    fp = feature_pipe(Xtr_raw)
    Xtr = pd.DataFrame(fp.fit_transform(Xtr_raw), index=Xtr_raw.index)
    Xte = pd.DataFrame(fp.transform(Xte_raw), index=Xte_raw.index)

    cv = PurgedWalkForwardCV(splits=8, gap=1)
    board = []

    # 1️⃣ classical & regularised linear
    for name in ["OLS","Ridge","Lasso","Elastic","BayesR","PCR","PLS"]:
        params = {"alpha":1.0} if name=="Ridge" else {}
        model = MODELS[name](params)
        board.append(dict(Model=name, **cv_score(model, Xtr, ytr, cv)))

    # 2️⃣ kernel / GP / tree / NN
    for name, params in [("SVR",{"C":1.0}), ("GP",{}), ("RF",{}),
                         ("LGB",{}), ("XGB",{}), ("Cat",{}),
                         ("NN",{})]:
        if name=="NN":
            model = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=SEED)
        else:
            model = MODELS[name](params)
        board.append(dict(Model=name, **cv_score(model, Xtr, ytr, cv)))

    # 3️⃣ stacked meta‑learner
    stacker=stacked().fit(Xtr,ytr)
    board.append(dict(Model="Stacked", **cv_score(stacker, Xtr, ytr, cv)))

    # leaderboard to CSV
    lb = pd.DataFrame(board).sort_values("RMSE")
    lb.to_csv("leaderboard.csv", index=False)
    logging.info("\n%s", lb.to_string(index=False))

    # choose top
    best_name = lb.iloc[0].Model
    best_model = stacker if best_name=="Stacked" else MODELS[best_name]({}).fit(Xtr, ytr)

    # hold‑out test
    yhat = best_model.predict(Xte)
    test_stats = reg_metrics(yte, yhat)
    logging.info("Hold‑out (%s) : %s", best_name, test_stats)

    # portfolio metrics
    port = port_metrics(pd.Series(yhat, index=yte.index), yte)
    logging.info("Portfolio metrics : %s", port)

    # save artefacts
    joblib.dump({"model":best_model, "feat_pipe":fp}, "tp_model.joblib")
    json.dump({**test_stats, **port}, open("test_report.json","w"), indent=2)

    try: shap_dump(best_model, pd.DataFrame(Xtr_raw))
    except Exception as e: logging.warning("SHAP failed: %s", e)

    # VAR benchmark
    var = VARBench().fit(train.drop(columns=[ycol]), ytr)
    var_pred = var.predict(test.drop(columns=[ycol]))
    logging.info("VAR RMSE %.4f", mean_squared_error(yte, var_pred, squared=False))

    logging.info("Finished in %s", datetime.now()-tic)

# ---------------------------------------------------------------------------
if __name__=="__main__":
    p=argparse.ArgumentParser(description="Ultimate linear → ensemble modeller")
    p.add_argument("--x", type=Path, required=True)
    p.add_argument("--y", type=Path, required=True)
    p.add_argument("--target", type=str, default="Y")
    p.add_argument("--holdout", type=int, default=2, help="years for OOS test")
    CFG=p.parse_args()
    run(CFG)
