from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import numpy as np

"""
## Regressors ###
Helper functions for common regression models. These can be used as fit_functions
in FailureModel.train(), or as examples for creating custom fit functions.

Expected signature for fit_function:
  Input: X (numpy array), Y (numpy array)
  Output: (model, metadata_dict) OR just model
  - model: trained model with a .predict(X) method
  - metadata_dict: dictionary with any relevant training information
"""

def fit_dummy(X, Y):
    model = DummyRegressor()
    model.fit(X.values, Y.values.ravel())
    return model

def fit_plsr(X, Y, optimal_components='best'):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('plsr', PLSRegression())
    ])
    X, Y = X.values, Y.values.ravel()
    if Y.shape[0] > 2:
        # Grid search over n_components
        max_n = max([2, min([20, X.shape[0]-1])])
        param_grid = {'plsr__n_components': list(range(1, max_n))}
        if Y.shape[0] < 5:
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=5)
        search = GridSearchCV(pipe, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        search.fit(X, Y)
        # Get mean and std of CV scores for each n_components
        cv_results = search.cv_results_
        n_components = cv_results['param_plsr__n_components'].data
        # Find the best score (highest, since scores are negative MSE)
        mean_test_scores = cv_results['mean_test_score']
        std_test_scores = cv_results['std_test_score']
        # remove nan
        mask_nan = np.isnan(mean_test_scores)
        n_components = n_components[~mask_nan]
        mean_test_scores = mean_test_scores[~mask_nan]
        std_test_scores = std_test_scores[~mask_nan]
        # 1-standard-error rule: pick simplest model within 1 std of best score
        best_idx = np.argmax(mean_test_scores)
        best_score = mean_test_scores[best_idx]
        best_std = std_test_scores[best_idx]
        threshold = best_score - best_std
        optimal_idx = np.where(mean_test_scores >= threshold)[0][0]

        if optimal_components == 'best':
            optimal_n_components = n_components[best_idx]
        elif optimal_components == '1se':
            optimal_n_components = n_components[optimal_idx]
        else:
            optimal_n_components = optimal_components
    else:
        optimal_n_components = 1

    # refit the model with optimal n_components, make test predictions
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('plsr', PLSRegression(n_components=optimal_n_components))
    ])
    # add mapie here
    model.fit(X, Y)
    return model

def fit_xgb(X, Y):
    pipe = Pipeline([
        ('xgb', XGBRegressor())
    ])
    # Grid search over n_components 1 to 10 using LOOCV
    param_grid = {
        'xgb__max_depth': np.array([2, 5, 10]),
        'xgb__min_child_weight': np.array([0.1, 0.5, 1]),
        'xgb__eta': np.array([0.05, 0.5, 1]),
    }
    if len(Y) < 5:
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=5)
    search = GridSearchCV(pipe, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    search.fit(X, Y)
    # make test predictions using the best modely
    return search.best_estimator_

def fit_plsr_featselect(X, Y, optimal_components='best', max_features=10):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_select', SelectFromModel(GradientBoostingRegressor(n_estimators=100), max_features=max_features)),
        ('plsr', PLSRegression())
    ])
    X, Y = X.values, Y.values.ravel()
    if Y.shape[0] > 2:
        # Grid search over n_components
        max_n = max([2, min([20, X.shape[0]-1])])
        param_grid = {'plsr__n_components': list(range(1, max_n))}
        if Y.shape[0] < 5:
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=5)
        search = GridSearchCV(pipe, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        search.fit(X, Y)
        # Get mean and std of CV scores for each n_components
        cv_results = search.cv_results_
        n_components = cv_results['param_plsr__n_components'].data
        # Find the best score (highest, since scores are negative MSE)
        mean_test_scores = cv_results['mean_test_score']
        std_test_scores = cv_results['std_test_score']
        # remove nan
        mask_nan = np.isnan(mean_test_scores)
        n_components = n_components[~mask_nan]
        mean_test_scores = mean_test_scores[~mask_nan]
        std_test_scores = std_test_scores[~mask_nan]
        # 1-standard-error rule: pick simplest model within 1 std of best score
        best_idx = np.argmax(mean_test_scores)
        best_score = mean_test_scores[best_idx]
        best_std = std_test_scores[best_idx]
        threshold = best_score - best_std
        optimal_idx = np.where(mean_test_scores >= threshold)[0][0]

        if optimal_components == 'best':
            optimal_n_components = n_components[best_idx]
        elif optimal_components == '1se':
            optimal_n_components = n_components[optimal_idx]
        else:
            optimal_n_components = optimal_components
    else:
        optimal_n_components = 1

    # refit the model with optimal n_components, make test predictions
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('plsr', PLSRegression(n_components=optimal_n_components))
    ])
    model.fit(X, Y)
    return model