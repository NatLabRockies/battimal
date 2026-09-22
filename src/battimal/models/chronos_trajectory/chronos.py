"Implement Chronos trajectory model for failure prediction"
from pathlib import Path
import pandas as pd
import numpy as np
from chronos import BaseChronosPipeline, Chronos2Pipeline

# eol_target = 'relative_discharge_capacity_C_5',
# is_eol_callable: callable = lambda pred_df, target: (pred_df[pred_df['target_name'] == target][pred_df.columns[4:]] < 0.8).all(axis=1).any())

class ChronosTrajectoryModel:
    def __init__(self, targets: list, eol_target: str, is_eol_callable: callable,
                 chronos_pipeline: BaseChronosPipeline = None,
                 id_column: str = 'cell_id', quantiles: list = [0.01, 0.99],
                 min_past: int = 3, prediction_len_percentile: float = 50,
                 finetune_kwargs: dict = None):
        
        if chronos_pipeline is None:
            # Load the Chronos-2# GPU recommended for faster inference, but CPU is also supported using device_map="cpu"
            chronos_pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(Path("chronos2_weights"), device_map="cpu")
        self.pipeline = chronos_pipeline
        
        self.targets = targets
        self.eol_target = eol_target
        self.is_eol_callable = is_eol_callable
        self.id_column = id_column
        self.quantiles = quantiles
        self.min_past = min_past
        self.prediction_len_percentile = prediction_len_percentile
        self.finetune_kwargs = finetune_kwargs

    # TODO
    def fit(self):
        return 
    
    # TODO
    def predict(self):
        return

    def assemble_chronos_inputs(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        summary_df['timestamp'] = summary_df['repeat_num'] * 1e14
        columns = self.targets + [self.id_column, 'timestamp']
        return summary_df[columns]

    def get_prediction_length(self, summary_df: pd.DataFrame) -> int:
        prediction_length = np.percentile(summary_df.groupby(self.id_column).size(), self.prediction_len_percentile) - self.min_past
        return max(0, prediction_length)

    def autoregress_to_failure(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        input_df = self.assemble_chronos_inputs(summary_df)
        # Initial prediction for all cells
        pred_df = self.pipeline.predict_df(
            input_df,
            prediction_length=self.get_prediction_length(summary_df),
            target=self.targets,
            id_column=self.id_column,
            quantile_levels=self.quantiles
        )

        pred_df_return = None
        for cell_id in pred_df[self.id_column].unique():
            # check if the cell has reached EOL based on the is_eol_callable function
            cell_pred_df = pred_df[pred_df[self.id_column] == cell_id]
            cell_input_df = input_df[input_df[self.id_column] == cell_id].copy()
            pred_cols = cell_pred_df.columns[3:].to_list()
            new_input_dfs = {
                    pred_col: cell_input_df.copy() for pred_col in pred_cols
                }
            while not self.is_eol_callable(cell_pred_df, self.eol_target):
                # If the cell has not reached EOL, need to autoregress the mean prediction and each quantile until we satisfy the end of life (eol) condition
                # To do this, for each of 'predictions' (mean prediction) and quantiles, append predicted targets to the input df for that cell, then re-run
                # the prediction with the updated input df for each quantile
                for pred_col in pred_cols:
                    # rearrange predictions from this cell and quantile to be formatted the same as the input df
                    cols = ['cell_id', 'target_name', 'timestamp'] + [pred_col]
                    cell_pred_df_copy = cell_pred_df[cols]
                    cell_pred_df_copy = cell_pred_df_copy.rename(columns={pred_col: 'predictions'})
                    cell_pred_df_copy = cell_pred_df_copy.set_index(['cell_id', 'target_name', 'timestamp'])
                    cell_pred_df_copy = cell_pred_df_copy.unstack('target_name')
                    cell_pred_df_copy = cell_pred_df_copy.reset_index() 
                    cell_pred_df_copy = pd.concat([cell_pred_df_copy['cell_id'], cell_pred_df_copy['timestamp'], cell_pred_df_copy['predictions']], axis=1)
                    # reformat timestamp back to int (same as input df)
                    cell_pred_df_copy['timestamp'] = cell_pred_df_copy['timestamp'].astype(int)

                    # append predictions to the input df
                    old_input_df = new_input_dfs[pred_col].copy()
                    new_cell_input_df = pd.concat([old_input_df, cell_pred_df_copy[cell_pred_df_copy['timestamp'] > old_input_df['timestamp'].max()]], axis=0).reset_index(drop=True)
                    new_input_dfs[pred_col] = new_cell_input_df.copy()

                    # Re-run the prediction with the updated input df
                    new_pred_df_this_quant = self.pipeline.predict_df(
                        new_cell_input_df,
                        prediction_length=self.get_prediction_length(summary_df),
                        target=self.targets,
                        id_column=self.id_column,
                        quantile_levels=self.quantiles
                    )
                    if pred_col == 'predictions':
                        new_pred_df = new_pred_df_this_quant
                    else:
                        new_pred_df[pred_col] = new_pred_df_this_quant[pred_col]
                # append the new predictions to the original pred_df
                cell_pred_df = cell_pred_df.set_index(['cell_id', 'target_name', 'timestamp'])
                new_pred_df = new_pred_df.set_index(['cell_id', 'target_name', 'timestamp'])
                cell_pred_df = pd.concat([cell_pred_df, new_pred_df], axis=0)
                cell_pred_df = cell_pred_df.reset_index()

            if pred_df_return is None:
                pred_df_return = cell_pred_df
            else:
                pred_df_return = pd.concat([pred_df_return, cell_pred_df], axis=0).reset_index(drop=True)

        return pred_df_return