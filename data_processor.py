import h5py
import time
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import DataFrame
from matplotlib import gridspec


class DataProcessor:
    def __init__(self):
        print("Process data for NASA's engines run-to-failure datasets")

    def load_hdf5_to_numpy_arr(self, file_name_hdf5):
        """
        Convert HDF5 raw data to numpy array.
        The data set is already broken to development and test sets. We just need to create separate variable sets for each.
        The dev set will be used for training the model and the test set will be used for evaluating and predictions.
        """

        op_time = time.process_time()

        with h5py.File(file_name_hdf5, "r") as hdf:

            # Get the model development set
            self.w_dev = np.array(hdf.get("W_dev"))  # Scenario-descriptor operating conditions
            self.x_s_dev = np.array(hdf.get("X_s_dev"))  # sensor measurements
            self.x_v_dev = np.array(hdf.get("X_v_dev"))  # Virtual sensors
            self.t_dev = np.array(hdf.get("T_dev"))  # Engine heath parameters
            self.y_rul_dev = np.array(hdf.get("Y_dev"))  # Target output Y - RUL of engine unit
            self.aux_dev = np.array(hdf.get("A_dev"))  # Auxiliary data

            # Get the model test set
            self.w_test = np.array(hdf.get("W_dev"))  # Scenario-descriptor operating conditions
            self.x_s_test = np.array(hdf.get("X_s_dev"))  # sensor measurements
            self.x_v_test = np.array(hdf.get("X_v_dev"))  # Virtual sensors
            self.t_test = np.array(hdf.get("T_dev"))  # Engine heath parameters
            self.y_rul_test = np.array(hdf.get("Y_dev"))  # Target output Y - RUL of engine unit
            self.aux_test = np.array(hdf.get("A_dev"))  # Auxiliary data

            # Get the variable names for each type of variable in the dataset
            self.w_var_names = np.array(hdf.get("W_var"))
            self.x_s_var_names = np.array(hdf.get("X_s_var"))
            self.x_v_var_names = np.array(hdf.get("X_v_var"))
            self.t_var_names = np.array(hdf.get("T_var"))
            self.aux_var_names = np.array(hdf.get("A_var"))

            # from np.array to list dtype U4/U5
            self.w_var_names = list(np.array(self.w_var_names, dtype="U20"))
            self.x_s_var_names = list(np.array(self.x_s_var_names, dtype="U20"))
            self.x_v_var_names = list(np.array(self.x_v_var_names, dtype="U20"))
            self.t_var_names = list(np.array(self.t_var_names, dtype="U20"))
            self.aux_var_names = list(np.array(self.aux_var_names, dtype="U20"))

        # Create complete development and test set of each varaible type
        self.w = np.concatenate((self.w_dev, self.w_test), axis=0)
        self.x_s = np.concatenate((self.x_s_dev, self.x_s_test), axis=0)
        self.x_v = np.concatenate((self.x_v_dev, self.x_v_test), axis=0)
        self.t = np.concatenate((self.t_dev, self.t_test), axis=0)
        self.y_rul = np.concatenate((self.y_rul_dev, self.y_rul_test), axis=0)
        self.aux = np.concatenate((self.aux_dev, self.aux_test), axis=0)

        # Generate dataframes
        self.df_aux = DataFrame(data=self.aux, columns=self.aux_var_names)

        self.df_t = DataFrame(data=self.t, columns=self.t_var_names)
        self.df_t["unit"] = self.df_aux["unit"].values
        self.df_t["cycle"] = self.df_aux["cycle"].values
        self.df_ts = self.df_t.drop_duplicates()

        self.df_w = DataFrame(data=self.w_dev, columns=self.w_var_names)
        self.df_w["unit"] = self.df_aux["unit"].values

        self.df_x_s = DataFrame(data=self.x_s, columns=self.x_s_var_names)
        self.df_v_s = DataFrame(data=self.x_v, columns=self.x_v_var_names)

        print("Operation time (sec): ", (time.process_time() - op_time))

    def custom_ts_multi_data_prep(x_data, y_data, start, end, window, horizon):
        """
        Create data used for model predictions.
        x_data: numpy array of input data
        y_data: numpy array of output (target) data
        start: start index data provided
        end: end index within data provided
        window: number of points in history as input to model
        horizon: number of points in future to predict the results
        """

        X = []
        y = []
        start = start + window
        if end is None:
            end = len(x_data) - horizon

        for i in range(start, end):
            indices = range(i - window, i)
            X.append(x_data[indices])

            indicey = range(i + 1, i + 1 + horizon)
            y.append(y_data[indicey])
        return np.array(X), np.array(y)
