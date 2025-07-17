import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def analyze_var_irf(X: pd.DataFrame, Y: pd.Series, maxlags: int = 12, ic: str = 'aic', horizon: int = 12):
    """
    Fit a VAR on [Y, X], compute orthogonalized IRFs, cumulative IRFs, FEVD, and summary metrics.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame of predictor variables (each column is one factor).
    Y : pd.Series
        Term premium series (with name attribute, e.g. Y.name = 'TP').
    maxlags : int
        Maximum lag order for VAR lag selection.
    ic : str
        Information criterion for lag selection ('aic', 'bic', 'hqic', 'fpe').
    horizon : int
        Number of periods ahead for IRF and FEVD.

    Returns
    -------
    dict
        {
            'var_results': fitted VARResults,
            'irf': IRAnalysis object,
            'cum_irf_df': DataFrame of cumulative IRFs,
            'fevd_df': DataFrame of FEVD shares,
            'summary_df': DataFrame of peak & decay metrics
        }
    """
    # Combine Y and X
    data = pd.concat([Y, X], axis=1)
    logger.info("Combined data shape: %s", data.shape)

    # Select lag order
    model = VAR(data)
    sel = model.select_order(maxlags=maxlags)
    selected_lag = int(getattr(sel, ic))
    logger.info("Selected lag (%s): %d", ic.upper(), selected_lag)

    # Fit VAR
    var_res = model.fit(selected_lag)
    logger.info("VAR fitted successfully")

    # Compute IRFs
    irf = var_res.irf(horizon)

    # Plot IRFs of Y to each X shock
    for factor in X.columns:
        fig = irf.plot(impulse=factor, response=Y.name, orth=True)
        fig.suptitle(f"IRF of {Y.name} to shock in {factor}", y=1.02)
        fig.tight_layout()
        plt.show()

    # Cumulative IRFs
    response_idx = data.columns.get_loc(Y.name)
    cum_irf = {
        factor: irf.irfs[:, response_idx, data.columns.get_loc(factor)].cumsum()
        for factor in X.columns
    }
    cum_irf_df = pd.DataFrame(cum_irf, index=np.arange(horizon+1))
    cum_irf_df.plot(title=f"Cumulative IRF of {Y.name}", xlabel="Horizon", ylabel="Cumulative Response")
    plt.tight_layout()
    plt.show()

    # FEVD
    fevd = var_res.fevd(horizon)
    fevd_decomp = fevd.decomp  # shape: (horizon, neqs, neqs)
    fevd_df = pd.DataFrame(
        {col: fevd_decomp[:, response_idx, i] for i, col in enumerate(data.columns)},
        index=np.arange(1, horizon+1)
    )
    fevd_df.plot(kind='bar', title=f"FEVD of {Y.name}", xlabel="Horizon", ylabel="Variance Share")
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

    # Summary metrics: peak and half-decay
    summary = []
    for factor in X.columns:
        irfs = irf.irfs[:, response_idx, data.columns.get_loc(factor)]
        peak_idx = int(np.argmax(np.abs(irfs)))
        peak_val = float(irfs[peak_idx])
        half_val = abs(peak_val) / 2
        decay_rel = np.where(np.abs(irfs[peak_idx:]) <= half_val)[0]
        decay_idx = int(peak_idx + decay_rel[0]) if decay_rel.size > 0 else np.nan
        summary.append({
            'factor': factor,
            'peak_horizon': peak_idx,
            'peak_value': peak_val,
            'half_decay_horizon': decay_idx
        })
    summary_df = pd.DataFrame(summary).set_index('factor')
    logger.info("IRF summary metrics:\n%s", summary_df)

    return {
        'var_results': var_res,
        'irf': irf,
        'cum_irf_df': cum_irf_df,
        'fevd_df': fevd_df,
        'summary_df': summary_df
    }

# Example usage:
# if __name__ == "__main__":
#     # Load your data
#     # X = pd.read_csv('factors.csv', index_col=0, parse_dates=True)
#     # Y = pd.read_csv('term_premium.csv', index_col=0, parse_dates=True)['TP']
#     # results = analyze_var_irf(X, Y, maxlags=12, ic='aic', horizon=12)
#     pass
