import h5py
import numpy as np
import pandas as pd

from sklearn import preprocessing
from pandas import DataFrame


class EngineData:
    def __init__(self):
        print("Process data for NASA's engines run-to-failure datasets")

    def load_hdf5_to_numpy_arr(self, file_name_hdf5):
        """
        Convert HDF5 raw data to numpy array.
        The data set is already broken to development and test sets. We just need to create separate variable sets for each.
        The dev set will be used for training the model and the test set will be used for evaluating and predictions.
        """

        with h5py.File(file_name_hdf5, "r") as hdf:

            # Get the model development set
            self.w_dev = np.array(hdf.get("W_dev"))  # Scenario-descriptor operating conditions
            self.x_s_dev = np.array(hdf.get("X_s_dev"))  # sensor measurements
            self.x_v_dev = np.array(hdf.get("X_v_dev"))  # Virtual sensors
            self.t_dev = np.array(hdf.get("T_dev"))  # Engine heath parameters
            self.y_rul_dev = np.array(hdf.get("Y_dev"))  # Target output Y - RUL of engine unit
            self.aux_dev = np.array(hdf.get("A_dev"))  # Auxiliary data

            # Get the model test set
            self.w_test = np.array(hdf.get("W_test"))  # Scenario-descriptor operating conditions
            self.x_s_test = np.array(hdf.get("X_s_test"))  # sensor measurements
            self.x_v_test = np.array(hdf.get("X_v_test"))  # Virtual sensors
            self.t_test = np.array(hdf.get("T_test"))  # Engine heath parameters
            self.y_rul_test = np.array(hdf.get("Y_test"))  # Target output Y - RUL of engine unit
            self.aux_test = np.array(hdf.get("A_test"))  # Auxiliary data

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

        # Create complete development and test set of each variable type
        self.w = np.concatenate((self.w_dev, self.w_test), axis=0)
        self.x_s = np.concatenate((self.x_s_dev, self.x_s_test), axis=0)
        self.x_v = np.concatenate((self.x_v_dev, self.x_v_test), axis=0)
        self.t = np.concatenate((self.t_dev, self.t_test), axis=0)
        self.y_rul = np.concatenate((self.y_rul_dev, self.y_rul_test), axis=0)
        self.aux = np.concatenate((self.aux_dev, self.aux_test), axis=0)

        # Generate dataframes
        self.df_aux = DataFrame(data=self.aux, columns=self.aux_var_names)
        self.df_x_s = DataFrame(data=self.x_s, columns=self.x_s_var_names)
        self.df_v_s = DataFrame(data=self.x_v, columns=self.x_v_var_names)
        self.df_t = DataFrame(data=self.t, columns=self.t_var_names)
        self.df_t["unit"] = self.df_aux["unit"].values
        self.df_t["cycle"] = self.df_aux["cycle"].values
        self.df_ts = self.df_t.drop_duplicates()
        self.df_w = DataFrame(data=self.w, columns=self.w_var_names)
        self.df_w["unit"] = self.df_aux["unit"].values

        self.generate_training_and_test_dataframes()

    def generate_training_and_test_dataframes(self):
        """ """
        self.df_rul_train = pd.DataFrame(data=self.y_rul_dev, columns=["RUL"])
        self.df_rul_test = pd.DataFrame(data=self.y_rul_test, columns=["RUL"])
        self.df_x_s_train = pd.DataFrame(data=self.x_s_dev, columns=self.x_s_var_names)
        self.df_x_s_test = pd.DataFrame(data=self.x_s_test, columns=self.x_s_var_names)
        self.df_x_v_train = pd.DataFrame(data=self.x_v_dev, columns=self.x_v_var_names)
        self.df_x_v_test = pd.DataFrame(data=self.x_v_test, columns=self.x_v_var_names)
        self.df_aux_test = pd.DataFrame(data=self.aux_test, columns=self.aux_var_names)
        self.df_aux_train = pd.DataFrame(data=self.aux_dev, columns=self.aux_var_names)
        self.df_w_test = pd.DataFrame(data=self.w_test, columns=self.w_var_names)
        self.df_w_train = pd.DataFrame(data=self.w_dev, columns=self.w_var_names)

        # We want to add Cycle, RUL and id to each dataframe category so we can study the data easier before model building and training
        self.df_x_s_train["cycle"] = self.df_aux_train["cycle"].values
        self.df_x_s_train["RUL"] = self.df_rul_train.values
        self.df_x_s_train["id"] = self.df_aux_train["unit"].values

        self.df_x_s_test["cycle"] = self.df_aux_test["cycle"].values
        self.df_x_s_test["RUL"] = self.df_rul_test.values
        self.df_x_s_test["id"] = self.df_aux_test["unit"].values

    def custom_ts_multi_data_prep(self, x_data, y_data, start, end, window, horizon):
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

    def gen_sequence(self, id_df, seq_length, seq_cols):
        """
        Generate network input sequence used for training.
        id_df : Dataframe for specific Engine Id
        seq_length : The window of time in history to be fed into the network
        seq_cols: Features in dataframe to be included in training
        """

        df_zeros = pd.DataFrame(np.zeros((seq_length - 1, id_df.shape[1])), columns=id_df.columns)
        id_df = df_zeros.append(id_df, ignore_index=True)
        data_array = id_df[seq_cols].values
        num_elements = data_array.shape[0]
        lstm_array = []

        for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
            lstm_array.append(data_array[start:stop, :])
        return np.array(lstm_array)

    def gen_label(self, id_df, seq_length, seq_cols, label):
        """
        Generate output labels that will be used for validation using training.
        id_df : Dataframe for specific Engine Id
        seq_length : The window of time in history to be fed into the network
        seq_cols: Features in dataframe to be included in training
        label: output label to be included in the y_dataset used for evaluation.
               this is essentially the label that model wll be trying to predict.
        """

        df_zeros = pd.DataFrame(np.zeros((seq_length - 1, id_df.shape[1])), columns=id_df.columns)
        id_df = df_zeros.append(id_df, ignore_index=True)
        data_array = id_df[seq_cols].values
        num_elements = data_array.shape[0]
        y_label = []

        for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
            y_label.append(id_df[label][stop])
        return np.array(y_label)

    def add_labels_to_dataset(
        self,
        data_frame,
        failure_in_w_cycles_1=30,
        failure_in_w_cycles_2=15,
        label_1_name="label1",
        label_2_name="label2",
    ):
        """
        Description: Add label columns for data frame
        "label1" is used binary classification, while trying to answer the question:
        is a specific engine going to fail within w1 cycles?
        """

        w1 = failure_in_w_cycles_1
        w0 = failure_in_w_cycles_2
        data_frame[label_1_name] = np.where(data_frame["RUL"] <= w1, 1, 0)
        data_frame[label_2_name] = data_frame[label_1_name]
        data_frame.loc[data_frame["RUL"] <= w0, label_2_name] = 2
        return data_frame

    def normalize_dataset(self, data_frame, features_to_exclude=["id", "cycle", "RUL", "label1", "label2"]):
        """
        Description: Normalize the data using MinMax normalization
        Inputs: List of features to exclude will ensure that data will not be normalized for those features.
        """

        data_frame["cycle_norm"] = data_frame["cycle"]
        cols_normalize = data_frame.columns.difference(features_to_exclude)
        min_max_scaler = preprocessing.MinMaxScaler()
        norm_train_df = pd.DataFrame(
            min_max_scaler.fit_transform(data_frame[cols_normalize]), columns=cols_normalize, index=data_frame.index
        )
        join_df = data_frame[data_frame.columns.difference(cols_normalize)].join(norm_train_df)
        data_frame = join_df.reindex(columns=data_frame.columns)
        return data_frame

    def generate_data_frame_for_specific_engine(self, data_frame, engine_id=1, horizon=0):
        """
        Description: Generate dataframe with specified engine id. If horizon is other than zero,
                     filter data with onl RUL that is less than the specified horizon since horizon.
        """

        df = data_frame[data_frame["id"] == engine_id]
        if horizon != 0:
            df[df["RUL"] <= df["RUL"].min() + horizon]
        return df

    def generate_lstm_x_y_inputs(
        self, data_frame, y_label="label1", x_columns_to_exclude=["RUL", "label1", "label2", "cycle", "id"], window=50
    ):
        """
        Keras LSTM layers expect an input in the shape of a numpy array of 3 dimensions (samples, time steps, features) = [N x T x D]
        where samples is the number of training sequences, time steps is the look back window or sequence length and features
        is the number of features of each sequence at each time step.
        """

        df_x = data_frame[data_frame.columns]
        df_y = data_frame.filter(y_label)
        df_x = df_x.drop(columns=x_columns_to_exclude)

        x = np.array(list(self.gen_sequence(df_x, window, df_x.columns)))
        y = np.array(list(self.gen_label(df_y, window, df_y.columns, y_label)))

        return (x, y)
