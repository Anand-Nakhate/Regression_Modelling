# src/models/RBPRegression.py

import pandas as pd
import numpy as np
from scipy.linalg import pinv
from src.models.base import BaseModel

class RBPRegression(BaseModel):
    """
    Relevance-Based Prediction (CKT) regression.
    For each date t, finds the optimal relevance threshold r* over a grid of percentiles,
    forms a relevance-weighted prediction of y_t, and records:
      – threshold (r*)
      – subsample size (n_obs)
      – subsample fraction (phi)
      – fit (squared corr. of weights & outcomes)
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, percentiles: np.ndarray = None):
        super().__init__()
        # --- Input checks & alignment ---
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise TypeError("X must be a pandas DataFrame or Series.")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series.")
        if not isinstance(X.index, pd.DatetimeIndex) or not isinstance(y.index, pd.DatetimeIndex):
            raise TypeError("Both X and y must have a DateTimeIndex.")
        merged = pd.concat([X, y.rename("target")], axis=1).dropna()
        self.X = merged.drop(columns="target").copy()
        self.y = merged["target"].copy()

        # --- Percentile grid for threshold search ---
        self.percentiles = np.asarray(percentiles) if percentiles is not None else np.linspace(0, 100, 11)

        # --- Placeholders for results ---
        self._is_fitted    = False
        self.fitted_values = None
        self.thresholds    = None
        self.n_obs         = None
        self.phi           = None
        self.fit           = None

    def fit(self) -> pd.Series:
        """
        Fit the RBP model in-sample.
        Returns
        -------
        pd.Series
            In-sample fitted values, indexed by self.X.index.
        """
        X = self.X.values
        y = self.y.values
        n, d = X.shape

        # 1) Precompute mean & inverse covariance for Mahalanobis
        mu      = X.mean(axis=0)
        cov     = np.cov(X, rowvar=False)
        inv_cov = pinv(cov)

        # 2) Precompute informativeness of each xi: Δ_i′ Σ⁻¹ Δ_i
        delta_i = X - mu
        info_i  = np.einsum("ij,jk,ik->i", delta_i, inv_cov, delta_i)

        # --- Containers for each date t ---
        thresholds, y_hats, n_obs, phis, fits = ([] for _ in range(5))

        for t in range(n):
            xt     = X[t]
            # similarity_i,t = −½·Mahalanobis(xi, xt)
            delta_xt = X - xt
            sim      = -0.5 * np.einsum("ij,jk,ik->i", delta_xt, inv_cov, delta_xt)

            # info_t = Mahalanobis(xt, μ)
            info_t = (xt - mu).dot(inv_cov).dot(xt - mu)

            # relevance scores r_i,t = sim + ½·(info_i + info_t)
            r = sim + 0.5 * (info_i + info_t)

            # search for best percentile threshold
            best = {"fit": -np.inf}
            for p in self.percentiles:
                r_star = np.percentile(r, p)
                mask   = r >= r_star
                k      = mask.sum()
                if k == 0:
                    continue

                phi     = k / n
                r_full  = r
                r_sub   = r[mask]
                var_full = np.mean((r_full - r_full.mean())**2)
                var_sub  = np.mean((r_sub  - r_sub.mean()) **2)
                lam      = np.sqrt(var_full/var_sub) if var_sub > 0 else 1.0

                # raw weights & normalization
                w_raw = np.zeros(n)
                w_raw[mask] = lam * (r_sub - phi)
                if w_raw.sum() == 0:
                    continue
                w = w_raw / w_raw.sum()

                # prediction & squared-corr fit
                y_pred = w.dot(y)
                if np.std(w) > 0 and np.std(y) > 0:
                    fit_t = np.corrcoef(w, y)[0,1]**2
                else:
                    fit_t = 0.0

                if fit_t > best["fit"]:
                    best = {
                        "fit":       fit_t,
                        "threshold": r_star,
                        "yhat":      y_pred,
                        "n_obs":     k,
                        "phi":       phi
                    }

            # record best for date t
            thresholds.append(best["threshold"])
            y_hats.append(best["yhat"])
            n_obs.append(best["n_obs"])
            phis.append(best["phi"])
            fits.append(best["fit"])

        # --- Store results as pandas Series ---
        idx = self.X.index
        self.fitted_values = pd.Series(y_hats,   index=idx, name="rbp_fitted")
        self.thresholds    = pd.Series(thresholds,index=idx, name="threshold")
        self.n_obs         = pd.Series(n_obs,     index=idx, name="n_obs")
        self.phi           = pd.Series(phis,      index=idx, name="phi")
        self.fit           = pd.Series(fits,      index=idx, name="fit")

        self._is_fitted = True
        return self.fitted_values

    def predict(self, X_new: pd.DataFrame = None) -> pd.Series:
        """
        Out-of-sample prediction on X_new. If X_new is None, returns in-sample fit.
        """
        if not self.is_fitted():
            print("Model not fit")
            return None

        # return in-sample if no new data
        if X_new is None:
            return self.fitted_values

        # align new data
        Xn = X_new.copy()
        if not set(self.X.columns).issubset(Xn.columns):
            raise ValueError("X_new must contain the same features as the training X.")
        Xn = Xn[self.X.columns]

        idx      = Xn.index
        X_train  = self.X.values
        y_train  = self.y.values
        n, _     = X_train.shape
        mu       = X_train.mean(axis=0)
        inv_cov  = pinv(np.cov(X_train, rowvar=False))
        info_i   = np.einsum("ij,jk,ik->i", (X_train-mu), inv_cov, (X_train-mu))

        preds = []
        for xt in Xn.values:
            # same loop as fit()
            sim    = -0.5 * np.einsum("ij,jk,ik->i", (X_train-xt), inv_cov, (X_train-xt))
            info_t = (xt-mu).dot(inv_cov).dot(xt-mu)
            r      = sim + 0.5 * (info_i + info_t)

            best_fit = -np.inf
            best_y   = None
            for p in self.percentiles:
                r_star = np.percentile(r, p)
                mask   = r >= r_star
                k      = mask.sum()
                if k == 0:
                    continue

                phi     = k / n
                var_full = np.mean((r - r.mean())**2)
                rs       = r[mask]
                var_sub  = np.mean((rs - rs.mean())**2)
                lam      = np.sqrt(var_full/var_sub) if var_sub > 0 else 1.0

                w_raw = np.zeros(n)
                w_raw[mask] = lam * (rs - phi)
                if w_raw.sum() == 0:
                    continue
                w = w_raw / w_raw.sum()

                y_pred = w.dot(y_train)
                if np.std(w) > 0 and np.std(y_train) > 0:
                    fit_t = np.corrcoef(w, y_train)[0,1]**2
                else:
                    fit_t = 0.0

                if fit_t > best_fit:
                    best_fit = fit_t
                    best_y   = y_pred

            preds.append(best_y)

        return pd.Series(preds, index=idx, name="rbp_predicted")

    def get_params(self) -> pd.DataFrame:
        """
        Return the full time series of thresholds, subsample sizes, fractions, and fits.
        """
        if not self.is_fitted():
            return None
        return pd.DataFrame({
            "threshold": self.thresholds,
            "n_obs":      self.n_obs,
            "phi":        self.phi,
            "fit":        self.fit
        })

    def get_metrics(self) -> dict:
        """
        In-sample metrics: MSE, R², and average per‐task fit.
        """
        if not self.is_fitted():
            return None
        resid = self.y - self.fitted_values
        mse   = float(np.mean(resid**2))
        sse   = np.sum(resid**2)
        sst   = np.sum((self.y - self.y.mean())**2)
        return {
            "mse":       mse,
            "rsquared":  float(1 - sse/sst),
            "mean_fit":  float(self.fit.mean())
        }

    def get_significance(self):
        """
        Not a standard param test, but expose all per-task stats.
        """
        if not self.is_fitted():
            return None
        return {
            "thresholds": self.thresholds,
            "n_obs":      self.n_obs,
            "phi":        self.phi,
            "fit":        self.fit
        }

    def is_fitted(self) -> bool:
        return getattr(self, "_is_fitted", False)

    def build_summary_table(self) -> pd.DataFrame:
        """
        Summarize the final date’s threshold, n_obs, phi, fit, plus MSE and R².
        """
        if not self.is_fitted():
            print("Model has not been fitted yet. Please run .fit() first.")
            return None

        mets = self.get_metrics()
        summary_df = pd.DataFrame({
            "threshold": [ self.thresholds.iloc[-1] ],
            "n_obs":     [ self.n_obs.iloc[-1] ],
            "phi":       [ self.phi.iloc[-1] ],
            "fit":       [ self.fit.iloc[-1] ],
            "mse":       [ mets["mse"] ],
            "rsquared":  [ mets["rsquared"] ]
        })
        return summary_df
