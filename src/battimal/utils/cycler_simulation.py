import numpy as np
import pandas as pd


"""
These classes handle processing data and stepping through the testing history of battery cells in a simulated cycler.
The key thing handled here is the data processing from each cell at each step, making it simple to directly compare the
performance of different types of models for predicting battery performance and end-of-life.

The structure of the various types of data is:
- summary_df: a dataframe with columns for cell_id, repeat_num, and any other relevant data collected at each cycle of testing
- failure_data: a tuple of (cell_ids, xy), where:
    - cell_ids is a list of lists of cell ids corresponding to the cell_ids for each row of x and y at each repeat_num
        - len(cell_ids) is the number of repeat_nums that have failure data
        - len(cell_ids[i]) is the number of cells that have failure data at repeat_num i
    - xy is a list of tuples of (x, y) dataframes, where x and y are the data for the failure prediction model at each repeat_num
        - len(xy) is the number of repeat_nums that have failure data
        - len(xy[i][0]) = len(xy[i][1]) = len(cell_ids[i]) is the number of cells that have failure data at repeat_num i
- rul_data: a tuple of (cell_ids, (x, y)), where:
    - cell_ids is a list of cell ids for all rows in x and y
    - x and y are dataframes for the remaining useful life prediction model
- lookahead_data: a tuple of (cell_ids, xy), which is similar to the failure_data but where the y dataframes are matrices of all future target values
- trajectory_data: a tuple of (cell_ids, xy), where:
    - cell_ids is a list of cell ids, one for each xy pair
    - xy is a list of tuples of (x, y) dataframes, one for each cell, where x and y are the data for the trajectory prediction model
- autoregressive_data: a tuple of (cell_ids, (x, y)), which is similar to the rul_data but where the x dataframes include past values of the target variable as features
"""

class Channel:
    """
    Represents a single channel of a battery cycler. Each channel is 'testing' one cell
    which progresses forward in time. Time can be represented by any variable. We know testing is done
    when either there's no data left or the cell has reached some end-of-life threshold.
    """
    def __init__(self, cell_id: int, 
                 summary_df: pd.DataFrame,
                 step_func: callable = lambda channel: channel,
                 eol_threshold_func: callable = lambda channel: False,
                 failure_data: tuple = None, rul_data: tuple = None, lookahead_data: tuple = None,
                 trajectory_data: tuple = None, autoregressive_data: tuple = None,
                 data_indexing_var: str = 'repeat_num', iter_step: int = 1):
        """
        Initializes the Channel with a cell ID, an optional end-of-life threshold function, and optional data matrices.
        Data matrices should be a tuple where entries are matrices of data for passing to different life models, i.e.,
        'failure', 'remaining_useful_life', etc.
        Args:
            cell_id (int): The ID of the cell being tested in this channel.
            summary_df (pd.DataFrame): Summary dataframe containing overall cell data.
            step_func (Callable[[Channel], Channel], optional): Function to execute for this channel at each step, returning the channel object. Defaults to a lambda function returning the channel unchanged.
            eol_threshold_func (Callable[[Channel], bool], optional): Function to determine end-of-life based on current data. Defaults to a lambda function returning False.
            failure_data (Optional[tuple]): Data for failure prediction models. Defaults to None.
            rul_data (Optional[tuple]): Data for remaining useful life models. Defaults to None.
            lookahead_data (Optional[tuple]): Data for lookahead models. Defaults to None.
            trajectory_data (Optional[tuple]): Data for trajectory prediction models. Defaults to None.
            autoregressive_data (Optional[tuple]): Data for autoregressive models. Defaults to None.
            data_indexing_var (str, optional): The variable to use for indexing the data from each source using the mask
                data[data_indexing_var] == iter. Defaults to 'repeat_num'.
            iter_step (int, optional): The step size to use when iterating through the data. Defaults to 1.
        """
    
        self.cell_id = cell_id
        # Functions
        self.step_func = step_func
        self.eol_threshold_func = eol_threshold_func
        self.is_eol = False
        # Data
        self.data_indexing_var = data_indexing_var
        self.iter_step = iter_step
        self.summary_df = summary_df
        self.failure_data = failure_data
        self.rul_data = rul_data
        self.lookahead_data = lookahead_data
        if lookahead_data is not None:
            self.lookahead_data_indices = np.array([int(self.lookahead_data[1][i][0][self.data_indexing_var].iloc[0]) for i in range(len(self.lookahead_data[1])) if len(self.lookahead_data[1][i][0]) > 0])
        else:
            self.lookahead_data_indices = None
        self.trajectory_data = trajectory_data
        self.autoregressive_data = autoregressive_data
        # Current data is a dictionary that gets updated at each step with the current data for this channel
        # collected by indexing each data type above at the current iteration
        self.current_data = {
            'summary_df': None,
            'failure_data': None,
            'rul_data': None,
            'lookahead_data': None,
            'trajectory_data': None,
            'autoregressive_data': None
        }
        # Initialize the channel by taking the first step
        self.iter = 0
        self.simple_iter = 0 # just counts steps one by one regardless of chronological variable and step size
        self.step() # updates self.current_data to iter 1 
    
    def step(self):
        """
        Steps forward one iteration, checking for EOL, updating the current data, and running the step function
        """
        self.iter += self.iter_step
        self.simple_iter += 1
        if self.iter >= self.summary_df.shape[0] or self.eol_threshold_func(self):
            self.is_eol = True
        self.update_data()
        self.step_func(self)

    def update_data(self) -> dict:
        """
        Returns the current data for this channel at the current iteration.
        Returns:
            dict: A dictionary containing the current data for the channel.
        """
        # Summary_df is a dataframe
        self.current_data['summary_df'] = self.summary_df[self.summary_df[self.data_indexing_var] == self.iter].reset_index(drop=True)
        if self.failure_data is not None and self.is_eol:
            self.current_data['failure_data'] = self.failure_data
        if self.rul_data is not None and self.is_eol:
            self.current_data['rul_data'] = self.rul_data
        if self.lookahead_data is not None:
            idx = np.where(self.lookahead_data_indices == self.iter)[0]
            if len(idx) > 0:
                idx = idx[0]
                self.current_data['lookahead_data'] = (self.lookahead_data[0][idx], self.lookahead_data[1][idx])
            else:
                self.current_data['lookahead_data'] = None
        if self.trajectory_data is not None:
            cell_id, (x, y) = self.trajectory_data
            self.current_data['trajectory_data'] = (cell_id, (x[self.iter-1:self.iter], y[self.iter-1:self.iter]))
        if self.autoregressive_data is not None:
            cell_id, (x, y) = self.autoregressive_data
            mask = x[self.data_indexing_var] == self.iter
            self.current_data['autoregressive_data'] = ([cell_id[0]], (x[mask], y[mask]))

class Cycler:
    """
    Simulates a battery cycler with multiple channels, each testing a different cell.
    """
    def __init__(self, n_channels: int, step_func: callable = lambda cycler: cycler, data_indexing_var: str = 'repeat_num', iter_step: int = 1):
        self.n_channels = n_channels
        self.channels = []
        self.n_empty_channels = n_channels
        self.num_total_measurements = 0
        self.num_total_cells = 0
        self.data_indexing_var = data_indexing_var
        self.iter_step = iter_step
        # Step function
        self.step_func = step_func
        # Current data is a dictionary that gets updated at each step with the current data for all channels past and present
        self.current_data = {
            'summary_df': None,
            'failure_data': ([[]], [([],[])]),
            'rul_data': None,
            'lookahead_data': ([], [([],[])]),
            'trajectory_data': None,
            'autoregressive_data': None
        }

    def add_channel(self, channel: Channel, idx: int = None):
        """
        Adds a channel to the cycler.
        Args:
            channel (Channel): The channel to add.
            idx (int): The channel idx to place that channel on. If idx is specified, can overwrite a filled channel.
        """
        if idx is not None and 0 <= idx < self.n_channels:
            self.channels[idx] = channel
        else:
            if self.n_empty_channels > 0:
                self.channels.append(channel)
            else:
                raise ValueError("Cycler is already at maximum channel capacity. Specific channel idx to replace an existing channel")
        self.n_empty_channels -= 1
        self.num_total_cells += 1

    def run_experiment(self, cell_order: list, summary_df: pd.DataFrame,
                       channel_step_func: callable = lambda channel: channel,
                       eol_threshold_func: callable = lambda channel: False,
                       failure_data: tuple = None, rul_data: tuple = None,
                       lookahead_data: tuple = None, trajectory_data: tuple = None,
                       autoregressive_data: tuple = None):
        """
        Runs the experiment until all cells in the cell_order reach end of life using the available channels. At each iteration,
        the cycler updates its current data and checks for end-of-life conditions for each channel. Any channel that reaches end-of-life
        is replaced with the next cell in the cell_order until there are no more cells to test. Any cell not at end-of-life steps forward
        each iteration, updating it's own current data and running the channel_step_func. The cycler also runs a step_func at each iteration
        Args:
            cell_order (list[int]): The order in which cells are tested in the channels.
            summary_df (pd.DataFrame): Summary dataframe containing overall cell data.
            channel_step_func (Optional[Callable[[Channel], Channel]]): Function to execute for each channel at each step, returning the channel object. Defaults to None.
            eol_threshold_func (Optional[Callable[[Channel], bool]]): Function to determine end-of-life based on current data. Defaults to None.
            failure_data (Optional[tuple]): Data for failure prediction models. Defaults to None.
            rul_data (Optional[tuple]): Data for remaining useful life models. Defaults to None.
            lookahead_data (Optional[tuple]): Data for lookahead models. Defaults to None.
            trajectory_data (Optional[tuple]): Data for trajectory prediction models. Defaults to None.
            autoregressive_data (Optional[tuple]): Data for autoregressive models. Defaults to None.
        """
        # Initialize channels with cells
        for i in range(min(self.n_channels, len(cell_order))):
            cell_id = cell_order.pop(0)
            cell_data = get_cell_data(cell_id, summary_df, failure_data, rul_data,
                                      lookahead_data, trajectory_data, autoregressive_data)
            channel = Channel(cell_id=cell_id,
                              eol_threshold_func=eol_threshold_func,
                              data_indexing_var=self.data_indexing_var,
                              iter_step=self.iter_step,
                              **cell_data)
            self.add_channel(channel)
        
        # Initialize the cycler's current data with the data from the initial channels
        self.update_current_data()
        # Accumulate total measurements and run the cycler step update function
        self.num_total_measurements += len(self.channels)
        self.step_func(self)

        # Run the experiment loop
        while len(cell_order) > 0 or not all([ch.is_eol for ch in self.channels]):
            # Check each channel for EOL and replace if there's more cells to test
            # Otherwise step forward in time for that channel (updating that channel's current data and running the channel step_func)
            for i, channel in enumerate(self.channels):
                if channel.is_eol:
                    if len(cell_order) > 0:
                        cell_id = cell_order.pop(0)
                        cell_data = get_cell_data(cell_id, summary_df, failure_data, rul_data,
                                                  lookahead_data, trajectory_data, autoregressive_data)
                        new_channel = Channel(cell_id=cell_id,
                                              step_func=channel_step_func,
                                              eol_threshold_func=eol_threshold_func,
                                              **cell_data)
                        self.add_channel(new_channel, i) # overwrites the existing cell on channel i
                    else:
                        self.channels.pop(i)
                        self.n_empty_channels += 1
                else:
                    channel.step()
            # Process the most recent data from each channel
            self.num_total_measurements += len(self.channels)
            self.update_current_data()
            # Run the cycler step update function
            self.step_func(self)
    
    def update_current_data(self):
        """
        Updates the current data for all channels in the cycler.
        """
        for channel in self.channels:
            channel_data = channel.current_data

            for data_type, data in channel_data.items():
                # 'data' is the new data from this channel for this data type (not including previous values)
                # Update the cycler data history for this type of data.
                #   If data is None, do nothing
                #   If cycler data[date_type] is None, simply write it
                #   Otherwise, update the cycler's current data with the new data from this channel, method depending on type of data
                if data_type == 'summary_df' and self.current_data[data_type] is not None:
                    data = pd.concat([self.current_data[data_type], data], ignore_index=True)
                    # Sort by cell_id and time (repeat_num) to keep like data grouped (could matter for complex models like TabPFN)
                    data = data.sort_values(by=['cell_id', 'repeat_num']).reset_index(drop=True)

                if data_type == 'failure_data' and data is not None:
                    # only appends when this cell has hit EOL
                    cell_ids, xy = self.current_data[data_type]
                    new_cell_ids, new_xy = data
                    for i in range(len(new_cell_ids)):
                        if i >= len(cell_ids):
                            # no previous cell has lasted this long, append new data
                            cell_ids.append(new_cell_ids[i])
                            xy.append(new_xy[i])
                        else:
                            cell_ids[i].extend(new_cell_ids[i])
                            if len(xy[i][0]) == 0:
                                # initial state, xy is empty, not a tuple
                                xy[i] = new_xy[i] # might just leave it empty if new_xy is also empty but that's fine
                            elif len(new_xy[i][0]) > 0:
                                # print('||||||||||||||||||||||||||||||||||||||||||||')
                                # print(i, xy[i])
                                # print('~~~~***************************************~~~')
                                # print(new_xy[i])
                                x, y = xy[i][0], xy[i][1]
                                x = pd.concat([x, new_xy[i][0]], ignore_index=True)
                                y = pd.concat([y, new_xy[i][1]], ignore_index=True)
                                xy[i] = (x, y)
                    data = (cell_ids, xy)
                    
                if data_type == 'lookahead_data' and data is not None:
                    cell_ids, xy = self.current_data[data_type]
                    new_cell_id, new_xy = data
                    # Add the new data aligned by the time index (iter) of the channel
                    if channel.simple_iter > len(cell_ids):
                        cell_ids.append(new_cell_id)
                        # special case where 1st measurement has been ignored
                        if len(xy) == 0:
                            while len(xy) < channel.simple_iter - 1:
                                xy.append([])
                        xy.append(new_xy)
                    else:
                        idx = channel.simple_iter - 1
                        cell_ids[idx].extend(new_cell_id)
                        x, y = xy[idx][0], xy[idx][1]
                        x = pd.concat([x, new_xy[0]], ignore_index=True)
                        y = pd.concat([y, new_xy[1]], ignore_index=True)
                        xy[idx] = (x, y)
                    data = (cell_ids, xy)

                if data_type == 'rul_data' and data is not None:
                    # only appends at eol
                    if self.current_data[data_type] is not None:
                        cell_ids, (x, y) = self.current_data[data_type]
                        new_cell_id, (new_x, new_y) = data
                        cell_ids.extend(new_cell_id)
                        x = pd.concat([x, new_x], ignore_index=True)
                        y = pd.concat([y, new_y], ignore_index=True)
                        data = (cell_ids, (x, y))

                if data_type == 'autoregressive_data' and data is not None:
                    if self.current_data[data_type] is not None:
                        cell_ids, (x, y) = self.current_data[data_type]
                        new_cell_id, (new_x, new_y) = data
                        # Add this data onto the existing data and resort to keep like data grouped (could matter for complex models like TabPFN)
                        cell_ids = cell_ids.append(new_cell_id)
                        x = pd.concat([x, new_x], ignore_index=True)
                        y = pd.concat([y, new_y], ignore_index=True)
                        # sort by cell_id and time (repeat_num)
                        sorted_indices = np.lexsort((x['repeat_num'], cell_ids))
                        cell_ids = [cell_ids[i] for i in sorted_indices]
                        x = x.iloc[sorted_indices].reset_index(drop=True)
                        y = y.iloc[sorted_indices].reset_index(drop=True)
                        data = (cell_ids, (x, y))

                if data_type == 'trajectory_data' and data is not None:
                    if self.current_data[data_type] is not None:
                        cell_ids, xy = self.current_data[data_type]
                        new_cell_id, new_xy = data
                        if new_cell_id in cell_ids:
                            idx = np.where(np.array(cell_ids) == new_cell_id)[0][0]
                            x, y = xy[idx][0], xy[idx][1]
                            x = pd.concat([x, new_xy[0]], ignore_index=True)
                            y = pd.concat([y, new_xy[1]], ignore_index=True)
                            xy[idx] = (x, y)
                        else:
                            cell_ids.append(new_cell_id)
                            xy.append(new_xy)
                        data = (cell_ids, xy)
                    else:
                        data = ([data[0]], [data[1]])
                
                if data is not None:
                    self.current_data[data_type] = data

def get_cell_data(cell_id: int, summary_df: pd.DataFrame,
                  failure_data: tuple = None, rul_data: tuple = None,
                  lookahead_data: tuple = None, trajectory_data: tuple = None,
                  autoregressive_data: tuple = None) -> dict:
    # index each data matrix / list of matrices by cell id
    cell_data = {}

    cell_data['summary_df'] = summary_df[summary_df['cell_id'] == cell_id].reset_index(drop=True)

    if failure_data is not None:
        cell_ids, xy = failure_data[0], failure_data[1]
        these_cell_ids = []
        these_xy = []
        for i, (cells, (x, y)) in enumerate(zip(cell_ids, xy)):
            mask_this_cell = [cell == cell_id for cell in cells]
            if any(mask_this_cell):
                last_idx = i
                these_cell_ids.append([cell_id])
                these_xy.append((x[mask_this_cell], y[mask_this_cell]))
            else:
                these_cell_ids.append([])
                these_xy.append(([], []))
        cell_data['failure_data'] = (these_cell_ids[:last_idx], these_xy[:last_idx])
    
    if rul_data is not None:
        cell_ids, (x, y) = rul_data[0], rul_data[1]
        cell_ids = pd.Series(cell_ids)
        mask_this_cell = cell_ids == cell_id
        cell_data['rul_data'] = (cell_ids[mask_this_cell].tolist(), (x[mask_this_cell], y[mask_this_cell]))

    if lookahead_data is not None:
        cell_ids, xy = lookahead_data[0], lookahead_data[1]
        these_cell_ids = []
        these_xy = []
        for i, (cells, (x, y)) in enumerate(zip(cell_ids, xy)):
            mask_this_cell = [cell == cell_id for cell in cells]
            if any(mask_this_cell):
                last_idx = i
                these_cell_ids.append([cell_id])
                these_xy.append((x[mask_this_cell], y[mask_this_cell]))
            else:
                these_cell_ids.append([])
                these_xy.append(([], []))
        cell_data['lookahead_data'] = (these_cell_ids[:last_idx], these_xy[:last_idx])

    if trajectory_data is not None:
        cell_ids, xy = trajectory_data[0], trajectory_data[1]
        idx_cell = int(np.where(np.array(cell_ids) == cell_id)[0][0])
        cell_data['trajectory_data'] = (cell_ids[idx_cell], xy[idx_cell])
    
    if autoregressive_data is not None:
        cell_ids, (x, y) = autoregressive_data[0], autoregressive_data[1]
        mask_this_cell = [cell == cell_id for cell in cell_ids]
        cell_data['autoregressive_data'] = (cell_ids[mask_this_cell], (x[mask_this_cell], y[mask_this_cell]))

    return cell_data
