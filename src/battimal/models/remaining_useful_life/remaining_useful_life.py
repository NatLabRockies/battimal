"""
Implements regressors that directly predict a remaining useful life (RUL) target from input features.

The RUL features and targets are a single matrix across the whole data set so training/testing is simple.
The RUL target needs to be transformed into the absolute lifetime before comparing with other models

Helper methods are built to directly implement PLSR and XGBoost regressors with hyperparameter tuning
"""

import numpy as np
import pandas as pd

class RemainingUsefulLifeModel:
    """
    A class for predicting remaining useful life using regression models.
    The actual training/inference of any model is extremely simple. Results
    need to be transformed into absolute lifetime for comparison with other models.
    
    Attributes:
        model: The regression model used for prediction.
        fit_function (callable): The function used to fit the regression model.
        prediction_transform (callable): A function to transform predictions to absolute lifetime.
                                         Assumed to have the signature: transform(x, y_prediction)
        only_numeric (bool): Whether to filter X to only numeric columns before training and inference.
    """

    def __init__(self, fit_function, prediction_transform=None, only_numeric=True):
        """
        Initialize the RemainingUsefulLifeModel.

        Args:
            fit_function (callable): The function used to fit the regression model.
            prediction_transform (callable, optional): A function to transform predictions to absolute lifetime.
                                                       Assumed to have the signature: transform(x, y_prediction)
            only_numeric (bool): Whether to filter X to only numeric columns before training and inference.
        """
        self.model = None
        self.fit_function = fit_function
        self.prediction_transform = prediction_transform
        self.only_numeric = only_numeric

    def train(self, X, Y):
        """
        Train the regression model.

        Args:
            X (DataFrame): The input features.
            Y (Series): The target variable.
        """
        if self.only_numeric:
            X = X.select_dtypes(include=[np.number])
        if len(X) >= 2:
            self.model = self.fit_function(X, Y)
        else:
            self.model = None

    def retrain(self, X, Y):
        """
        Same as 'train', provided for API consistency.

        Args:
            X (DataFrame): The input features.
            Y (Series): The target variable.
        """
        self.train(X, Y)
    
    def predict(self, X, return_transformed=False):
        """
        Predict the remaining useful life.

        Args:
            X (DataFrame): The input features.
            return_transformed (bool): Whether to return the transformed predictions.

        Returns:
            Series: The predicted remaining useful life.
        """
        if self.only_numeric:
            X = X.select_dtypes(include=[np.number])

        if self.model is not None:
            predictions = self.model.predict(X)
            if self.prediction_transform is not None and return_transformed:
                predictions = self.prediction_transform(predictions)
        else:
            predictions = np.full(len(X), np.nan)

        return predictions

"""
Prediction transform:
    y target is 'log_remaining_weeks' so remaining weeks is exp(y)
    and then absolute lifetime is x['num_week'] + exp(y)
"""
def log_rul_to_lifetime(x, y, idx_var='repeat_num'):
    remaining_weeks = np.exp(y)
    absolute_lifetime = x[idx_var] + remaining_weeks
    return absolute_lifetime

def rul_to_lifetime(x, y, idx_var='repeat_num'):
    absolute_lifetime = x[idx_var] + y
    return absolute_lifetime