# var_models.py — state‑of‑the‑art VAR & TVP‑VAR modelling toolkit
from __future__ import annotations
import json, hashlib, logging, warnings, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import structlog
import optuna
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tools.eval_measures import aic, bic

SEED = 42
np.random.seed(SEED)
warnings.filterwarnings("ignore")

###############################################################################
# Logging
###############################################################################
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    processors=[structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                structlog.processors.JSONRenderer()]
)
log = structlog.get_logger()

def run_id() -> str:
    return hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8]

###############################################################################
# Metrics
###############################################################################
def reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    mae  = np.abs(y_true - y_pred).mean()
    hit  = (np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))).mean()
    return dict(RMSE=rmse, MAE=mae, HitRatio=hit)

###############################################################################
# 1.  CONSTANT‑PARAMETER VAR
###############################################################################
def fit_var(train: pd.DataFrame, max_lags: int = 12) -> tuple[VAR, int]:
    sel = VAR(train).select_order(max_lags)
    p   = sel.aic  # use AIC; switch to sel.bic to penalise harder
    model = VAR(train).fit(p)
    log.info("var_fitted", lags=p)
    return model, p

def rolling_var_forecast(df: pd.DataFrame, test_size: int, max_lags: int = 12):
    preds, actuals = [], []
    for split in range(test_size):
        train = df.iloc[: -test_size + split]
        test_point = df.iloc[-test_size + split]
        model, _ = fit_var(train, max_lags)
        preds.append(model.forecast(train.values[-model.k_ar:], 1)[0])
        actuals.append(test_point.values)
    preds = np.vstack(preds)
    actuals = np.vstack(actuals)
    return preds, actuals

###############################################################################
# 2.  TIME‑VARYING‑PARAMETER VAR  (TVP‑VAR)
###############################################################################
class TVPVAR(MLEModel):
    """
    Implements a TVP‑VAR(1) with diffuse initialisation.
    For brevity we fix p = 1; extendible by augmenting the state.
    """
    def __init__(self, endog: pd.DataFrame):
        k_endog = endog.shape[1]
        super().__init__(endog, k_states=k_endog ** 2, initialization='diffuse')
        self.k_endog = k_endog

        # Observation Eq.: y_t = Z_t * beta_t
        # where Z_t = kron(I_k, y_{t-1}')  (1 × k^2)
        self['design'] = np.zeros((k_endog, k_endog ** 2, self.nobs))
        for t in range(1, self.nobs):
            z_t = np.kron(np.eye(k_endog), endog.iloc[t - 1].values)
            self.ssm['design', :, :, t] = z_t

        # State Eq.: beta_t = beta_{t-1} + eta_t, eta_t ~ N(0, Q)
        self['transition'] = np.eye(k_endog ** 2)
        self['selection'] = np.eye(k_endog ** 2)

        # Initialise covariance matrices to be estimated
        self['state_cov'] = np.eye(k_endog ** 2) * 0.01     # Q
        self['obs_cov']   = np.eye(k_endog)                 # H

    def update(self, params, **kwargs):
        # params[0] = log σ² for obs noise; params[1] = log σ² state noise
        obs_var, state_var = np.exp(params)
        self['obs_cov']   = np.eye(self.k_endog) * obs_var
        self['state_cov'] = np.eye(self.k_endog ** 2) * state_var

    @property
    def start_params(self):
        return np.log([0.1, 0.001])   # log‑variance initial guesses

    def transform_params(self, unconstrained):
        return unconstrained          # exp() applied in update()

def fit_tvpvar(train: pd.DataFrame):
    mod = TVPVAR(train)
    res = mod.fit(disp=False)
    log.info("tvpvar_fitted",
             llf=res.llf,
             obs_sigma=np.exp(res.params[0]),
             state_sigma=np.exp(res.params[1]))
    return res

def tvpvar_one_step_forecasts(res, test: pd.DataFrame) -> np.ndarray:
    """Generate 1‑step‑ahead forecasts from Kalman prediction."""
    preds = []
    last_state = res.filtered_state[:, -1]
    last_cov   = res.filtered_state_cov[:, :, -1]

    for i in range(len(test)):
        z_t = np.kron(np.eye(res.model.k_endog), test.iloc[i - 1].values)
        forecast_mean = z_t @ last_state
        preds.append(forecast_mean)

        # Kalman prediction step for next β_t
        last_state = last_state       # random walk: mean unchanged
    return np.vstack(preds)

###############################################################################
# 3.  Hyper‑parameter search for TVP‑VAR state noise (optional)
###############################################################################
def tune_tvpvar(train: pd.DataFrame):
    def objective(trial):
        sigma_q = trial.suggest_loguniform("sigma_q", 1e-6, 1e-1)
        mod = TVPVAR(train)
        mod['state_cov'] *= sigma_q / 0.01
        res = mod.fit(disp=False)
        return -res.llf   # maximise likelihood
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    best_q = study.best_params["sigma_q"]
    log.info("tvpvar_tuned", sigma_q=best_q)
    return best_q

###############################################################################
# 4.  Main orchestration
###############################################################################
def main(args):
    rid = run_id()
    arte_dir = Path(f"artefacts_{rid}").mkdir(exist_ok=True)
    df = (pd.read_parquet(args.data) if args.data.suffix in {".pq", ".parquet"}
          else pd.read_csv(args.data, index_col=0, parse_dates=True))
    df = df.ffill().dropna()

    test_size = args.test
    train, test = df.iloc[:-test_size], df.iloc[-test_size:]

    # ---------------- VAR ----------------
    var_preds, var_actuals = rolling_var_forecast(df, test_size, max_lags=12)
    var_metrics = reg_metrics(var_actuals[:, df.columns.get_loc(args.target)],
                              var_preds[:, df.columns.get_loc(args.target)])
    log.info("var_metrics", **var_metrics)
    json.dump(var_metrics, open(Path(arte_dir) / "var_metrics.json", "w"), indent=2)

    # save IRF and FEVD on full‑sample VAR
    var_full, p_opt = fit_var(df)
    irf = var_full.irf(12)
    fevd = var_full.fevd(12)
    irf.plot(orth=False).figure.savefig(Path(arte_dir) / "irf.png", dpi=150)
    fevd.plot().figure.savefig(Path(arte_dir) / "fevd.png", dpi=150)
    var_full.save(Path(arte_dir) / "var_results.pkl")

    # --------------- TVP‑VAR -------------
    sigma_q = tune_tvpvar(train)  # comment this line to skip tuning
    tvp_model = TVPVAR(train)
    tvp_model['state_cov'] *= sigma_q / 0.01
    tvp_res = tvp_model.fit(disp=False)
    tvp_preds = tvpvar_one_step_forecasts(tvp_res, df.iloc[-test_size-1:-1])

    tvp_metrics = reg_metrics(var_actuals[:, df.columns.get_loc(args.target)],
                              tvp_preds[:, df.columns.get_loc(args.target)])
    log.info("tvpvar_metrics", **tvp_metrics)
    json.dump(tvp_metrics, open(Path(arte_dir) / "tvpvar_metrics.json", "w"), indent=2)

    # store state paths for diagnostics
    beta_paths = pd.DataFrame(tvp_res.smoothed_state.T,
                              index=train.index, columns=[
                                  f"β_{i}" for i in range(tvp_res.smoothed_state.shape[0])
                              ])
    beta_paths.to_parquet(Path(arte_dir) / "tvp_coefficients.parquet")

    log.info("finished", artefacts=str(arte_dir))

###############################################################################
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="Parquet/CSV with datetime index & all endogenous vars")
    ap.add_argument("--target", type=str, default="TP",
                    help="Column for metric evaluation")
    ap.add_argument("--test", type=int, default=24,
                    help="# observations in hold‑out (e.g., 24 months)")
    main(ap.parse_args())
