import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from typing import Dict, Sequence


class NelsonSiegelSvenssonFitter:
    """
    Fits the Nelson–Siegel–Svensson yield‑curve model:
        y(τ) = β0
             + β1 * ((1 - e^{-τ/τ1}) / (τ/τ1))
             + β2 * [((1 - e^{-τ/τ1}) / (τ/τ1)) - e^{-τ/τ1}]
             + β3 * [((1 - e^{-τ/τ2}) / (τ/τ2)) - e^{-τ/τ2}]
    where τ is time to maturity.
    """

    def __init__(
        self,
        tau1_bounds: Sequence[float] = (1e-6, 30.0),
        tau2_bounds: Sequence[float] = (1e-6, 60.0),
    ):
        # bounds for the nonlinear parameters τ1 and τ2
        self._bounds = ([tau1_bounds[0], tau2_bounds[0]], [tau1_bounds[1], tau2_bounds[1]])

    @staticmethod
    def _design_matrix(maturities: np.ndarray, tau1: float, tau2: float) -> np.ndarray:
        """
        Build the  (n_obs x 4) design matrix for linear β's, given τ1 and τ2.
        """
        t = maturities
        # avoid division by zero
        t1 = np.maximum(tau1, 1e-6)
        t2 = np.maximum(tau2, 1e-6)
        # factors
        f1 = (1 - np.exp(-t / t1)) / (t / t1)
        f2 = f1 - np.exp(-t / t1)
        f3 = (1 - np.exp(-t / t2)) / (t / t2) - np.exp(-t / t2)

        # [1, f1, f2, f3]
        return np.column_stack([np.ones_like(t), f1, f2, f3])

    def _residuals_for_tau(self, tau: Sequence[float], t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Given tau = [τ1, τ2], compute residuals y_model - y_data, 
        where β are the OLS solution for that tau.
        """
        tau1, tau2 = tau
        X = self._design_matrix(t, tau1, tau2)
        # linear OLS to get β's
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return X.dot(beta) - y

    def fit(self, maturities: np.ndarray, yields: np.ndarray) -> Dict[str, float]:
        """
        Fit one cross‑section: maturities and yields arrays (same length, with no NaNs).
        Returns dict with keys ['beta0','beta1','beta2','beta3','tau1','tau2'].
        """
        # initial guess for [τ1, τ2]
        x0 = [2.0, 10.0]

        # nonlinear solve on τ's
        sol = least_squares(
            fun=self._residuals_for_tau,
            x0=x0,
            bounds=self._bounds,
            args=(maturities, yields),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )

        tau1_opt, tau2_opt = sol.x
        # final β's
        X_opt = self._design_matrix(maturities, tau1_opt, tau2_opt)
        beta_opt, *_ = np.linalg.lstsq(X_opt, yields, rcond=None)

        return {
            "beta0": float(beta_opt[0]),
            "beta1": float(beta_opt[1]),
            "beta2": float(beta_opt[2]),
            "beta3": float(beta_opt[3]),
            "tau1": float(tau1_opt),
            "tau2": float(tau2_opt),
        }

    def fit_series(
        self,
        df: pd.DataFrame,
        col_to_maturity: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Iterate over each date (row) in `df`, fit only on columns
        where the yield is not null, and return a DataFrame of parameters.
        
        Parameters
        ----------
        df : pd.DataFrame
            index = dates, columns = various yield series (must match keys of col_to_maturity)
        col_to_maturity : dict
            maps df.columns names -> time‑to‑maturity in years
        
        Returns
        -------
        pd.DataFrame
            index = same dates as input df;
            columns = ['beta0','beta1','beta2','beta3','tau1','tau2']
        """
        records = []
        for date, row in df.iterrows():
            # select maturities & yields where not null
            valid = row[col_to_maturity.keys()].dropna()
            if len(valid) < 4:
                # need at least 4 points to pin down 4 betas
                records.append({**{"beta0": np.nan, "beta1": np.nan,
                                   "beta2": np.nan, "beta3": np.nan,
                                   "tau1": np.nan, "tau2": np.nan},
                                 "date": date})
                continue

            t = np.array([col_to_maturity[col] for col in valid.index], dtype=float)
            y = valid.values.astype(float)

            params = self.fit(t, y)
            params["date"] = date
            records.append(params)

        result = pd.DataFrame(records).set_index("date").sort_index()
        return result


if __name__ == "__main__":
    # === example usage ===
    # df: your DataFrame, e.g.
    #         'C01': 0.5%, 'C02': 1.2%, 'C05': 2.1%, ...   (column names arbitrary)
    # mapping = {'C01': 0.25, 'C02': 0.5, 'C05': 2.0, ...}  # in years
    import pandas as pd

    # load or assemble your df here
    # df = pd.read_csv("yields.csv", index_col=0, parse_dates=True)
    # define your mapping col_name -> maturity
    # mapping = {col: float(years) for col, years in ...}

    fitter = NelsonSiegelSvenssonFitter()
    nss_params_df = fitter.fit_series(df, mapping)

    print(nss_params_df.head())
