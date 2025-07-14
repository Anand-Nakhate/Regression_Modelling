import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

# ─────────────────────────────────────────
# Logging
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
warnings.simplefilter('ignore', ConvergenceWarning)

# ─────────────────────────────────────────
# 1) Classical VAR
# ─────────────────────────────────────────
def fit_var(data: pd.DataFrame,
            maxlags: int = 15,
            ic: str = 'aic') -> Tuple[VAR, pd.DataFrame]:
    """
    Fit a standard Vector Autoregression (VAR) model.

    Parameters
    ----------
    data : pd.DataFrame
      Multivariate time series with columns = variables.
    maxlags : int
      Maximum number of lags to consider.
    ic : str
      Information criterion for lag selection ('aic','bic','hqic','fpe').

    Returns
    -------
    results : VARResults
      Fitted VAR model results.
    forecast_df : pd.DataFrame
      One-step-ahead in-sample fitted values.
    """
    model = VAR(data)
    logging.info("Selecting lag order up to %d by %s", maxlags, ic.upper())
    lag_order = model.select_order(maxlags)
    p_selected = getattr(lag_order, ic)
    logging.info("Selected p = %d", p_selected)

    res = model.fit(p_selected)
    fitted = pd.DataFrame(
        res.fittedvalues,
        index=data.index[p_selected:],
        columns=data.columns
    )
    return res, fitted

# ─────────────────────────────────────────
# 2) TVP-VAR via Kalman Filter
# ─────────────────────────────────────────
class TVPVAR:
    """
    Time-Varying Parameter VAR(p) implemented via Kalman Filter.
    State vector = stacked VAR coefficients (K * p * K dimensions).
    Observation: y_t = Z_t * beta_t + eps_t,
    State eq: beta_t = beta_{t-1} + w_t (random walk)
    """
    def __init__(self,
                 p: int = 1,
                 Q_scale: float = 1e-5,
                 R_scale: float = 1e-2,
                 ):
        self.p = p
        self.Q_scale = Q_scale
        self.R_scale = R_scale
        self.k = None  # number of series
        self.kf = None
        self.beta_smoothed = None

    def _build_design(self, Y: np.ndarray) -> np.ndarray:
        # Y shape: (T, k)
        T, k = Y.shape
        Ks = k * self.p * k
        Z = np.zeros((T, k, Ks))
        for t in range(self.p, T):
            x_stack = []
            for lag in range(1, self.p+1):
                x_stack.append(Y[t - lag])
            x_stack = np.concatenate(x_stack)
            # design row: kron(I_k, x_stack^T)
            Z[t] = np.kron(np.eye(k), x_stack[np.newaxis, :])
        return Z  # shape (T, k, Ks)

    def fit(self, data: pd.DataFrame):
        Y = data.values
        T, k = Y.shape
        self.k = k
        Ks = k * self.p * k

        # Build time-varying design matrices
        Z = self._build_design(Y)

        # Initialize KalmanFilter
        kf = KalmanFilter(
            k_endog = k,
            k_states = Ks,
            transition = np.eye(Ks),
            selection = np.eye(Ks),
            state_cov = self.Q_scale * np.eye(Ks),
            design = Z,
            obs_cov = self.R_scale * np.eye(k)
        )
        # Provide initial state and covariance
        init_state = np.zeros(Ks)
        init_cov = np.eye(Ks) * 1.0

        kf.bind(Y)
        kf.initialize_known(init_state, init_cov)

        # Run Kalman smoother
        logging.info("Running Kalman smoother for TVP-VAR(p=%d)", self.p)
        smooth_res = kf.smooth()

        # extract smoothed states: shape (T, Ks)
        self.beta_smoothed = smooth_res.smoothed_state.T
        self.kf = kf
        return self

    def get_time_varying_coefs(self) -> np.ndarray:
        """
        Returns: beta[t] for t in p..T-1, shape = (T-p, k, k*p)
        """
        T = self.beta_smoothed.shape[0]
        Ks = self.k * self.p * self.k
        reshaped = self.beta_smoothed.reshape(T, self.k, self.p * self.k)
        return reshaped[self.p:]

    def forecast(self,
                 data: pd.DataFrame,
                 steps: int = 1) -> pd.DataFrame:
        # Perform one-step forecast using last smoothed state
        last_beta = self.beta_smoothed[-1]
        coefs = last_beta.reshape(self.k, self.p * self.k)
        x_last = []
        for lag in range(1, self.p+1):
            x_last.append(data.values[-lag])
        x_stack = np.concatenate(x_last)
        yhat = coefs.dot(x_stack)
        idx = pd.date_range(start=data.index[-1], periods=steps+1, freq=data.index.freq)[1:]
        return pd.DataFrame([yhat], index=idx, columns=data.columns)

# ─────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────
if __name__ == "__main__":
    # load your monthly dataset
    df = pd.read_csv("/path/to/monthly_data.csv", index_col=0, parse_dates=True)

    # -------- VAR --------
    var_res, var_fit = fit_var(df, maxlags=12, ic='aic')
    print(var_res.summary())

    # one-step ahead forecast
    var_fc = var_res.forecast(df.values[-var_res.k_ar:], steps=3)
    print("VAR Forecast:\n", pd.DataFrame(var_fc, columns=df.columns))

    # -------- TVP-VAR --------
    tvp = TVPVAR(p=2, Q_scale=1e-5, R_scale=1e-2)
    tvp.fit(df)
    # extract time-varying coefficients
    betas = tvp.get_time_varying_coefs()
    print("TVP-VAR coefs shape:", betas.shape)

    # forecast
    tvp_fc = tvp.forecast(df, steps=3)
    print("TVP-VAR Forecast:\n", tvp_fc)
