"""
Implements regressors that directly predict a failure target from input features, with each
iterative set of inputs (weeks, cycles, RPTs, ...) treated as independent train/test sets.

Failure matrices will come as lists of tuples of x and y data from independent sets, which
are paired with lists of the cell_ids corresponding to the cells of each row of X and Y in each set.

Helper methods are built to directly implement PLSR and XGBoost regressors with hyperparameter tuning
"""

import numpy as np
import pandas as pd

class FailureModel:
    """
    A class for failure prediction models that treats each iterative set of inputs 
    (weeks, cycles, RPTs, etc.) as independent train/test sets.
      Attributes:
        models (list): List of trained models, one per independent set (None for failed training)
        fit_function (callable): Function used to train models
        prediction_transform (callable): Function to transform predictions (e.g., inverse scaling)
                                         Assumed to have the signature: transform(x, y_prediction)
        is_chain_regressor (bool): Whether to use chain regression between sets
        idx_valid_sets (list): List of set indices that failed to train
        only_numeric (bool): Whether to filter X to only numeric columns before training and inference.
    """

    def __init__(self, fit_function, prediction_transform=None, is_chain_regressor=False, only_numeric=True):
        """
        Initialize the FailureModel.
          Parameters:
            fit_function (callable): Function that takes (X, Y) DataFrames and returns (model, metadata_dict).
                                    The function should train a model on the input data and return
                                    both the trained model and a dictionary of metadata about the training.
                                    Can also return just the model, in which case basic metadata is created.
                                    The model is assumed to have a sklearn-type API, where we can make inferences
                                    using the trained model object via 'model.predict(X)'.
                                    Example: lambda X, Y: fit_plsr(X, Y, optimal_components='1se')
            prediction_transform (callable, optional): Function to transform predictions (e.g., inverse scaling).
                                    Assumed to have the signature: transform(x, y_prediction)
            is_chain_regressor (bool): Whether to use chain regression to propogate past predictions to future sets.
                                    The chain regressor uses the untransformed predictions from the prior set.
            only_numeric (bool): Whether to filter X to only numeric columns before training.
        """
        if not callable(fit_function):
            raise ValueError("fit_function must be callable")
        
        self.fit_function = fit_function
        self.prediction_transform = prediction_transform
        self.models = []
        self.idx_valid_sets = []
        self.is_chain_regressor = is_chain_regressor
        self.only_numeric = only_numeric

    def train(self, cell_ids, xy_pairs):
        """
        Train a list of models, one per independent set.
        Parameters:
            cell_ids (list): List of cell_id lists corresponding to the rows in X and Y of each set.
            xy_pairs (list): List of tuples (X, Y), where X and Y are DataFrames
        
        Returns:
            self: Returns the instance of self with trained models
        """
        
        self.models = []
        self.is_valid_set = []
        self.cell_ids_set = []
        self.prior_prediction_by_cell = None

        for cell_id, (X, Y) in zip(cell_ids, xy_pairs):
            # Check for insufficient data
            if len(X) < 2:
                self.models.append(None)
                self.is_valid_set.append(False)
                self.cell_ids_set.append([])
            else:
                if self.only_numeric:
                    X = X.select_dtypes(include=[np.number])

                # Chain regressor: add each cell's last valid prediction as feature
                if self.is_chain_regressor and self.prior_prediction_by_cell is not None:    
                    X['prior_Y_pred'] = [self.prior_prediction_by_cell[cell] for cell in cell_id]

                # Fit and store model object, track input cells to model
                model = self.fit_function(X, Y)
                self.models.append(model)
                self.is_valid_set.append(True)
                self.cell_ids_set.append(cell_id)

                # Chain regressor: first trained model instantiates and fills prior_predictions_by_cell
                # Later iterations update prior_prediction_by_cell
                if self.is_chain_regressor:
                    Y_predict = model.predict(X.values)
                    if self.prior_prediction_by_cell is None:
                        self.prior_prediction_by_cell = {cell: y for cell, y in zip(cell_id, Y_predict)}
                    else:
                        self.prior_prediction_by_cell.update({cell: y for cell, y in zip(cell_id, Y_predict)})

        return self

    def predict(self, cell_ids, x, set_number=None, return_transformed=False):
        """
        Make predictions using the trained models. Applies the prediction_transform if specified.
        
        Parameters:
            cell_ids (list or list of lists): List of cell_id values for each row in X, REQUIRED if is_chain_regressor
                                             because we need to look up previous predictions but unused otherwise.
            x (DataFrame or list): Input features for prediction.
                                       Can be a single DataFrame (with set_number specified) or a list of 
                                       DataFrames (one per model) whose indices correspond to the model indices.
            set_number (int, optional): If x is a single DataFrame, specify which model to use (0-indexed).
                                       If None, x should be a list of dataframes, assumed to be index matched
                                       with the corresponding model index (x[i] used with model[i]). When
                                       using a chain regression approach, assumes that the prior predictions from the
                                       most recently trained model have already been made and stored (which is done
                                       automatically by calling predict)
            return_transformed (bool): Whether to return transformed predictions only. If False, returns raw predictions.

        Returns:
            predictions (array or list): Predictions. If set_number is specified, returns a single array.
                                        Otherwise, returns a list of arrays (one per set).
                                        For models that failed to train, returns array of NaNs with appropriate shape.
        """
        if not self.models:
            raise ValueError("No models have been trained yet. Call train() first.")
        if self.is_chain_regressor and cell_ids is None:
            raise ValueError("cell_ids must be provided when using chain regression.")
        
        if x.__class__ != list:
            x = [x]
            cell_ids = [cell_ids]
            if set_number is None:
                raise ValueError("When x is a single DataFrame, set_number must be specified.")
            if set_number < 0 or set_number >= len(self.models):
                raise ValueError(f"set_number {set_number} is out of range. Must be 0-{len(self.models)-1}")
            flag_return_list = False
        else:
            if set_number is not None:
                raise UserWarning("When x is a list, set_number is ignored.")
            if len(x) != len(cell_ids):
                raise ValueError(f"Length of x ({len(x)}) and cell_ids ({len(cell_ids)}) must match.")
            set_number = range(min(len(x), len(self.models)))
            flag_return_list = True

        if self.is_chain_regressor and 'prior_predictions_by_cell' not in self.__dict__:
            self.prior_predictions_by_cell = None

        predictions = []
        for idx, this_x, cell_id in zip(set_number, x, cell_ids):
            model = self.models[idx]
            if self.only_numeric:
                this_x = this_x.select_dtypes(include=[np.number])

            # Return NaNs if model failed to train
            if model is None:
                predictions.append(np.full(len(this_x), np.nan))
                continue

            if self.is_chain_regressor and idx > 0:
                if self.prior_predictions_by_cell is None:
                    raise KeyError("Can only use chain regressor if prior predictions exist!")
                this_x['prior_Y_pred'] = [self.prior_predictions_by_cell[cell] for cell in cell_id]

            y_predict = model.predict(this_x.values)

            if self.is_chain_regressor:
                # Update prior_predictions_by_cell for future use
                if self.prior_predictions_by_cell is None:
                    self.prior_predictions_by_cell = {cell: y for cell, y in zip(cell_id, y_predict)}
                else:
                    self.prior_predictions_by_cell.update({cell: y for cell, y in zip(cell_id, y_predict)})

            if self.prediction_transform is not None and return_transformed:
                y_predict = self.prediction_transform(this_x, y_predict)
            predictions.append(y_predict)

        if flag_return_list:
            return predictions
        else:
            return predictions[0]
    
    def get_residuals(self, cell_ids, xy_pairs, y_predictions):
        """
        Get a DataFrame with all residuals. Each has the true value, predicted value, residual, cell_id, and set number.
        
        Parameters:
            cell_ids (list): List of cell_id lists corresponding to the rows in X and Y of each set.
            xy_pairs (list): List of tuples (X, Y), where X and Y are DataFrames.
            y_predictions (list): List of predicted values for each set.
        Returns:            residuals_df (DataFrame): DataFrame with columns ['cell_id', 'set_number', 'Y_true', 'Y_pred', 'residual']

        """
        residuals = []
        for cell_id_set, (X, Y), y_pred in zip(cell_ids, xy_pairs, y_predictions):
            for cell_id, true, pred in zip(cell_id_set, Y.values, y_pred):
                if self.prediction_transform is not None:
                    true_transformed = self.prediction_transform(X, [true])[0]
                    pred_transformed = self.prediction_transform(X, [pred])[0]
                    residuals.append({
                        'cell_id': cell_id,
                        'set_number': len(residuals),
                        'Y_true': true,
                        'Y_pred': pred,
                        'residual': true - pred,
                        'Y_true_transformed': true_transformed,
                        'Y_pred_transformed': pred_transformed,
                        'residual_transformed': true_transformed - pred_transformed
                    })
                else:
                    residuals.append({
                        'cell_id': cell_id,
                        'set_number': len(residuals),
                        'Y_true': true,
                        'Y_pred': pred,
                        'residual': true - pred
                    })
        return pd.DataFrame(residuals)
