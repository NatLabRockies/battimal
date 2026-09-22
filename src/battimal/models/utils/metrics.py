import numpy as np
import pandas as pd

### Metrics
def get_metrics(residuals, metrics_functions, filter_nan=False, group_by=None, n_bins=10):
    """
    Compute metrics from a residuals DataFrame. If the residuals DataFrame contains transformed
    predictions (columns 'Y_true_transformed' and 'Y_pred_transformed'), metrics will also be
    computed on the transformed values, with results prefixed by 'transformed_'.
    
    Parameters:
        residuals (DataFrame): Residuals DataFrame with at least columns 'Y_true', 'Y_pred', 'residual'.
                                Optionally contains 'Y_true_transformed', 'Y_pred_transformed', 'residual_transformed'.
                                May also contain 'set_number' and/or 'cell_id' for grouping.
        metrics_functions (callable or list of callables): Metric function(s) to apply.
                                Each metric function should have signature: metric(y_true, y_pred) -> float.
        filter_nan (bool): Whether to filter out NaN/inf predicted values before calculating metrics.
                            If False, any NaN in predictions will result in NaN metric.
                            Default is False.
        group_by (str, optional): Column name to group residuals by before computing metrics.
                                    Valid options: 'set_number', 'cell_id', 'Y_true', or None.
                                    If None (default), metrics are computed across all residuals (single value).
                                    If 'set_number' or 'cell_id', groups by unique values of that column.
                                    If 'Y_true', bins Y_true values into n_bins equal-width bins (since Y_true
                                    is assumed to be continuous numeric).
        n_bins (int): Number of bins to use when group_by='Y_true'. Default is 10.

    Returns:
        metrics (dict): Dictionary mapping metric names to their computed values.
                        If group_by is None, each value is a single float.
                        If group_by is specified, each value is a DataFrame with columns
                        [group_by, <metric_name>] - one row per group.
                        For group_by='Y_true', the group_by column contains the bin midpoints.
    """
    if not isinstance(metrics_functions, list):
        metrics_functions = [metrics_functions]
    
    # Validate group_by
    valid_group_by = [None, 'set_number', 'cell_id', 'Y_true']
    if group_by not in valid_group_by:
        raise ValueError(f"group_by must be one of {valid_group_by}, got {group_by!r}")
    if group_by is not None and group_by not in residuals.columns:
        raise ValueError(f"group_by column {group_by!r} not found in residuals DataFrame")
    
    # Determine if we have transformed residuals
    has_transformed = 'Y_true_transformed' in residuals.columns and 'Y_pred_transformed' in residuals.columns
    
    # If grouping by Y_true (continuous), bin the values
    residuals_grouped = residuals
    group_col = group_by
    if group_by == 'Y_true':
        residuals_grouped = residuals.copy()
        # Create equal-width bins on Y_true; use bin midpoints as the group label
        bin_edges = np.linspace(residuals['Y_true'].min(), residuals['Y_true'].max(), n_bins + 1)
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        # include_lowest ensures the minimum value is included
        bin_indices = pd.cut(
            residuals['Y_true'], bins=bin_edges, labels=False, include_lowest=True
        )
        residuals_grouped['_Y_true_bin'] = bin_midpoints[bin_indices.astype(int)]
        group_col = '_Y_true_bin'
    
    def _compute_metric(metric, y_true, y_pred):
        """Helper to compute a single metric with optional NaN filtering."""
        y_true = np.asarray(y_true).astype(float)
        y_pred = np.asarray(y_pred).astype(float)
        if filter_nan:
            y_pred = y_pred.copy()
            y_pred[np.isinf(y_pred)] = np.nan
            mask = ~(np.isnan(y_pred) | np.isnan(y_true))
            if np.sum(mask) == 0:
                return np.nan
            return metric(y_true[mask], y_pred[mask])
        else:
            if np.any(np.isnan(y_pred)) or np.any(np.isnan(y_true)):
                return np.nan
            return metric(y_true, y_pred)
    
    metrics = {}
    for metric in metrics_functions:
        name = metric.__name__
        
        if group_by is None:
            # Compute single metric across all residuals
            metrics[name] = _compute_metric(metric, residuals['Y_true'], residuals['Y_pred'])
            if has_transformed:
                metrics["transformed_" + name] = _compute_metric(
                    metric, residuals['Y_true_transformed'], residuals['Y_pred_transformed']
                )
        else:
            # Compute metric per group, return as DataFrame
            group_values = []
            metric_values = []
            transformed_values = [] if has_transformed else None
            
            for group_val, subset in residuals_grouped.groupby(group_col, sort=True):
                group_values.append(group_val)
                metric_values.append(_compute_metric(metric, subset['Y_true'], subset['Y_pred']))
                if has_transformed:
                    transformed_values.append(
                        _compute_metric(metric, subset['Y_true_transformed'], subset['Y_pred_transformed'])
                    )
            
            metrics[name] = pd.DataFrame({group_by: group_values, name: metric_values})
            if has_transformed:
                transformed_name = "transformed_" + name
                metrics[transformed_name] = pd.DataFrame({
                    group_by: group_values,
                    transformed_name: transformed_values
                })
    
    return metrics

def _nlpd_expression(y, mu, var):
    z = (y - mu) ** 2 / var + np.log(2 * np.pi * var)
    return z

def compute_nlpd(truth_y, predicted_y):
    """
    Negative log predictive density. Uses the definition in
    https://jmlr.org/papers/volume17/15-327/15-327.pdf
    """

    if np.isscalar(truth_y) and predicted_y.ndim == 1:
        # Case for scalar truth_y and 1D predicted_y
        mu = np.mean(predicted_y)
        var = np.var(predicted_y)
        nlpd = 0.5 * (_nlpd_expression(truth_y, mu, var) + np.log(2 * np.pi * var))
    else:
        T = predicted_y.shape[1]
        nlpd = 0.0
        for i in range(T):
            mu = np.mean(predicted_y[:, i])
            var = np.var(predicted_y[:, i])
            nlpd += _nlpd_expression(truth_y[i], mu, var)
        nlpd = 0.5 * nlpd / T

    return nlpd