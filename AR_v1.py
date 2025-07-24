#!/usr/bin/env python3
# ar_models.py — Ultimate ARIMA & ARIMAX forecasting

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------------------------------------------------------------------
# CONFIG & LOGGING
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
)
SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------------------------
# DATA LOADER
# ------------------------------------------------------------------------------
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

    def load_panel(self, target_col: str, shift_target: bool = False) -> pd.DataFrame:
        """Join X and Y on the index; optionally shift Y for t+1 forecasting."""
        df = self.x_df.join(
            self.y_df.rename(columns={self.y_df.columns[0]: target_col}),
            how='inner'
        )
        if shift_target:
            df[target_col] = df[target_col].shift(-1)
        df = df.ffill().dropna()
        logging.info("Data loaded: %d obs × %d cols", *df.shape)
        return df

# ------------------------------------------------------------------------------
# ARIMA/ARIMAX via pmdarima.auto_arima
# ------------------------------------------------------------------------------
class ARIMAModel:
    def __init__(self, seasonal_period: int = 0):
        """
        seasonal_period > 1 turns on seasonal ARIMA with period = seasonal_period.
        """
        self.seasonal_period = seasonal_period
        self.model: pm.arima.ARIMA = None

    def grid_search(
        self,
        y: pd.Series,
        exog: pd.DataFrame | None = None,
        max_p: int = 5, max_d: int = 2, max_q: int = 5,
        max_P: int = 2, max_D: int = 1, max_Q: int = 2,
        trace: bool = True
    ) -> pm.arima.ARIMA:
        seasonal = self.seasonal_period > 1
        m = self.seasonal_period if seasonal else 1

        self.model = pm.auto_arima(
            y=y,
            exogenous=exog,
            seasonal=seasonal,
            m=m,
            max_p=max_p, max_d=max_d, max_q=max_q,
            max_P=max_P, max_D=max_D, max_Q=max_Q,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            trace=trace,
            random_state=SEED
        )
        logging.info("Auto-ARIMA selected: %s", self.model.summary())
        return self.model

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> pm.arima.ARIMA:
        if self.model is None:
            raise RuntimeError("Call grid_search() first.")
        self.model.fit(y, exogenous=exog)
        return self.model

    def predict(
        self,
        n_periods: int,
        exog: pd.DataFrame | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        fc, conf = self.model.predict(
            n_periods=n_periods,
            exogenous=exog,
            return_conf_int=True
        )
        return fc, conf

    def summary(self) -> str:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        return str(self.model.summary())

# ------------------------------------------------------------------------------
# SARIMAX/ARIMAX via statsmodels with grid-search (by AIC)
# ------------------------------------------------------------------------------
class SARIMAXModel:
    def __init__(
        self,
        order: tuple[int,int,int] = (1,0,0),
        seasonal_order: tuple[int,int,int,int] = (0,0,0,0)
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.results: sm.regression.linear_model.RegressionResultsWrapper = None

    def fit(
        self,
        endog: pd.Series,
        exog: pd.DataFrame | None = None
    ) -> sm.regression.linear_model.RegressionResultsWrapper:
        mod = SARIMAX(
            endog,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.results = mod.fit(disp=False)
        logging.info(
            "SARIMAX fitted: order=%s seasonal=%s AIC=%.2f",
            self.order, self.seasonal_order, self.results.aic
        )
        return self.results

    def forecast(
        self,
        steps: int,
        exog: pd.DataFrame | None = None
    ) -> tuple[pd.Series, pd.DataFrame]:
        if self.results is None:
            raise RuntimeError("Call fit() first.")
        fc_obj = self.results.get_forecast(steps=steps, exog=exog)
        return fc_obj.predicted_mean, fc_obj.conf_int()

    def summary(self) -> str:
        if self.results is None:
            raise RuntimeError("Model not fitted.")
        return str(self.results.summary())

# ------------------------------------------------------------------------------
# UTILITIES: metrics & plotting
# ------------------------------------------------------------------------------
def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": mean_squared_error(actual, predicted, squared=False),
        "MAE": mean_absolute_error(actual, predicted)
    }

def plot_forecast(
    history: pd.Series,
    fc_vals: np.ndarray,
    conf_int: np.ndarray,
    title: str,
    save_path: Path
):
    plt.figure(figsize=(10,6))
    plt.plot(history.index, history, label="Observed")
    # build future index
    freq = history.index.freq or pd.infer_freq(history.index)
    last = history.index[-1]
    future_idx = pd.date_range(last, periods=len(fc_vals)+1, freq=freq)[1:]
    plt.plot(future_idx, fc_vals, label="Forecast")
    plt.fill_between(
        future_idx,
        conf_int[:, 0],
        conf_int[:, 1],
        color='b', alpha=0.2
    )
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    logging.info("Saved plot: %s", save_path)

# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Ultimate ARIMA & SARIMAX Forecasting"
    )
    p.add_argument("--x", required=True, help="Features CSV/Parquet path")
    p.add_argument("--y", required=True, help="Target CSV/Parquet path")
    p.add_argument("--target", default="Y", help="Name of target column")
    p.add_argument(
        "--seasonal_period", type=int, default=0,
        help="Seasonal period (e.g. 12); <=1 → no seasonality"
    )
    p.add_argument(
        "--exog", action="store_true",
        help="Treat all X cols as exogenous variables"
    )
    p.add_argument(
        "--forecast_periods", type=int, default=12,
        help="Number of steps to forecast"
    )
    args = p.parse_args()

    # load data
    loader = DataLoader(Path(args.x), Path(args.y))
    df = loader.load_panel(target_col=args.target, shift_target=False)
    y = df[args.target]
    exog = df.drop(columns=[args.target]) if args.exog else None

    # ---- 1) Auto-ARIMA / ARIMAX (pmdarima) ----
    arima = ARIMAModel(seasonal_period=args.seasonal_period)
    logging.info("Running auto_arima grid search…")
    arima.grid_search(y, exog=exog)
    arima.fit(y, exog=exog)
    fc1, ci1 = arima.predict(
        n_periods=args.forecast_periods,
        exog=(exog[-args.forecast_periods:] if args.exog else None)
    )
    plot_forecast(
        y, fc1, ci1,
        title="Auto-ARIMA Forecast",
        save_path=Path("auto_arima_forecast.png")
    )

    # ---- 2) Manual SARIMAX grid-search by AIC + final fit ----
    # parameter grid
    p = d = q = range(0, 3)
    pdq = [(i,j,k) for i in p for j in d for k in q]
    P = D = Q = range(0, 2)
    if args.seasonal_period > 1:
        seasonal_pdq = [
            (i, j, k, args.seasonal_period)
            for i in P for j in D for k in Q
        ]
    else:
        seasonal_pdq = [(0,0,0,0)]

    best_aic = np.inf
    best_cfg = None
    best_res = None
    logging.info("Searching SARIMAX grid by AIC…")
    for order in pdq:
        for sod in seasonal_pdq:
            try:
                tmp = SARIMAX(
                    y,
                    exog=exog,
                    order=order,
                    seasonal_order=sod,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)
                if tmp.aic < best_aic:
                    best_aic, best_cfg, best_res = tmp.aic, (order, sod), tmp
            except Exception:
                continue

    logging.info("Best SARIMAX %s AIC=%.2f", best_cfg, best_aic)
    sar = SARIMAXModel(order=best_cfg[0], seasonal_order=best_cfg[1])
    sar.results = best_res

    fc2, ci2 = sar.forecast(
        steps=args.forecast_periods,
        exog=(exog[-args.forecast_periods:] if args.exog else None)
    )
    plot_forecast(
        y, fc2, ci2,
        title="Manual SARIMAX Forecast",
        save_path=Path("sarimax_forecast.png")
    )

    # ---- 3) Compare metrics ----
    actual = y.iloc[-args.forecast_periods :].values
    m1 = compute_metrics(actual, fc1)
    m2 = compute_metrics(actual, fc2)
    comp = pd.DataFrame([m1, m2], index=["AutoARIMA", "SARIMAX"])
    comp.to_csv("forecast_metrics.csv")
    logging.info("Saved metrics to forecast_metrics.csv\n%s", comp)

    print("\n===== Forecast Performance =====")
    print(comp)

if __name__ == "__main__":
    main()
