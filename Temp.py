import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

class RollingOLSRegression(BaseModel):
    …

    def plot_fitted_with_ci(self, alpha: float = 0.05, figsize=(10,6)):
        """
        Plot actual Y vs. rolling‐OLS fitted values,
        with (1−alpha)*100% confidence bands.
        """
        if not self.is_fitted():
            raise RuntimeError("You must call .fit() before plotting.")

        # 1) Extract fit results
        #    params: DataFrame, shape (T, k)
        params = self.results.params  
        #    cov_params: 3D array of shape (T, k, k)
        cov_stack = self.results.cov_params  

        # 2) Fitted values at each time (right‐edge of window)
        fitted = self.results.fittedvalues  # Series length T

        # 3) Build arrays for lower/upper bands
        lower = np.empty_like(fitted)
        upper = np.empty_like(fitted)

        # Number of regressors (including const if present)
        k = params.shape[1]
        # Degrees of freedom for t‐crit: window − #params
        df = self.window - k
        tcrit = t.ppf(1 - alpha/2, df)

        # 4) For each timestamp, compute se of prediction
        #    x_t must align with the same ordering used in params/cov
        #    Here we assume self.X columns exactly match params.columns
        Xmat = self.X[params.columns].values  # shape (T, k)
        for i in range(len(fitted)):
            x_t = Xmat[i]                  # 1×k
            cov_t = cov_stack[i]           # k×k
            se_fit = np.sqrt(x_t @ cov_t @ x_t)
            lower[i] = fitted.iloc[i] - tcrit * se_fit
            upper[i] = fitted.iloc[i] + tcrit * se_fit

        # Turn into pandas.Series
        lower = pd.Series(lower, index=fitted.index)
        upper = pd.Series(upper, index=fitted.index)

        # 5) Plot everything
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.Y.index, self.Y, label="Actual", linewidth=1)
        ax.plot(fitted.index, fitted, label="Fitted", linewidth=1)
        ax.fill_between(
            fitted.index, 
            lower, 
            upper, 
            alpha=0.3, 
            label=f"{int((1-alpha)*100)}% CI"
        )
        ax.set_title("Rolling OLS: Actual vs Fitted with Confidence Bands")
        ax.set_xlabel("Date")
        ax.set_ylabel(self.Y.name or "Y")
        ax.legend(loc="upper left")
        plt.tight_layout()

        return fig, ax
