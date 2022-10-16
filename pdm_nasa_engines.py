"""
    Project Name: PdM Model for NASA Jet Engines
    Data: 10/9/2022
    Author: Arpi Derm
    Description: This model aims to use PdM model and RUL to predict failures in point time T in near future.
                 This specific model is built and trained for the NASA's PCoE Turbofan Engine Degradation (Run-To-Failure) dataset.
    Dataset Description: 
                Dataset description along with its variable can be found in the dataset paper writtent by 
                Manuel Arias Chao,Chetan Kulkarni, Kai Goebel and Olga Fink. https://dx.doi.org/10.3390/data6010005
    Variable Name Descriptions:
            w = Scenario-descriptor operating conditions (inputs to system model)
            x_s = sensor signal measurements (physical properties)
            x_v = virtual sensor signals
            t = engine health parameters 
            y_rul = target output Y - RUL (Remaning Useful Life) of engine unit
            aux = Auxiliaryl data such as cycle count, unit id, ...
"""

from tabnanny import verbose
from data_processor import DataProcessor
from visualizer_analyzer import DataAV
from google.colab import drive

drive.mount("/content/drive")

# Load all datasets
DS_001 = "/content/drive/MyDrive/Predictive_Maintenence_Fault_Detection/data_set/N-CMAPSS_DS01-005.h5"
DS_002 = "/content/drive/MyDrive/Predictive_Maintenence_Fault_Detection/data_set/N-CMAPSS_DS02-006.h5"
DS_003 = "/content/drive/MyDrive/Predictive_Maintenence_Fault_Detection/data_set/N-CMAPSS_DS03-012.h5"
DS_004 = "/content/drive/MyDrive/Predictive_Maintenence_Fault_Detection/data_set/N-CMAPSS_DS04.h5"
DS_005 = "/content/drive/MyDrive/Predictive_Maintenence_Fault_Detection/data_set/N-CMAPSS_DS05.h5"
DS_006 = "/content/drive/MyDrive/Predictive_Maintenence_Fault_Detection/data_set/N-CMAPSS_DS06.h5"
DS_007 = "/content/drive/MyDrive/Predictive_Maintenence_Fault_Detection/data_set/N-CMAPSS_DS07.h5"
DS_008 = "/content/drive/MyDrive/Predictive_Maintenence_Fault_Detection/data_set/N-CMAPSS_DS08a-009.h5"
DS_009 = "/content/drive/MyDrive/Predictive_Maintenence_Fault_Detection/data_set/N-CMAPSS_DS08c-008.h5"
DS_010 = "/content/drive/MyDrive/Predictive_Maintenence_Fault_Detection/data_set/N-CMAPSS_DS08d-010.h5"


class EngineRUL:
    def __init__(self, dataset):

        # Load data set and prepare data frames
        self.pros = DataProcessor()
        self.pros.load_hdf5_to_numpy_arr(dataset)

        # Create a visualizer class. This is useful for understanding the data we work with
        # it will determine feature extraction and model architecture to some extend
        self.vs_an = DataAV(
            self.pros.df_aux,
            self.pros.df_x_s,
            self.pros.x_s_var_names,
            self.pros.df_w,
            self.pros.df_t,
            self.pros.t_var_names,
            self.pros.w_var_names,
            self.pros.df_ts,
        )

    def visualize_data(self):
        """
        Generate all visualization graphs and plots necessary for understanding data.
        """

        self.vs_an.get_engine_units_in_dataset()
        self.vs_an.plot_flight_classes()
        self.vs_an.show_engine_health_parameter_stats()
        self.vs_an.generate_engine_health_parameter_graphs()
        self.vs_an.generate_hpt_eff_over_cycles_all_engines()
        self.vs_an.generate_sensor_readings_graphs_single_unit(1)
        self.vs_an.generate_sensor_readings_graphs_single_unit_single_cycle(1, 2)
        self.vs_an.plot_health_states_for_all_engines()
        self.vs_an.generate_flight_profle_single_unit_single_cycle(1, 2)
        self.vs_an.generate_flight_envelope()
        self.vs_an.generate_kde_estimations_of_flight_profile()

    def generate_tf_data_for_model(self, window, horizon, train_split, batch_size, buffer_size):
        """
        Genearate the tf dataset with proper dimensionality and shape given input parameters.
        This dataset will split into training and validation subsets.
        """

        self.x_train, self.y_train = self.pros.custom_ts_multi_data_prep(
            self.pros.w, self.pros.y_rul, 0, train_split, window, horizon
        )

        self.x_vali, self.y_vali = self.pros.custom_ts_multi_data_prep(
            self.pros.w, self.pros.y_rul, train_split, None, window, horizon
        )

        self.train_data = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.train_data = self.train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()

        self.val_data = tf.data.Dataset.from_tensor_slices((self.x_vali, self.y_vali))
        self.val_data = self.val_data.batch(batch_size).repeat()

    def generate_pdm_model(self):
        """
        Build the DNN model with its layers that will be used for RUL preditcions.
        """

        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(200, return_sequences=True), input_shape=self.x_train.shape[-2:]
                ),
                tf.keras.layers.Dense(20, activation="tanh"),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
                tf.keras.layers.Dense(20, activation="tanh"),
                tf.keras.layers.Dense(20, activation="tanh"),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(1),
            ]
        )

        self.model.compile(optimizer="adam", loss="rmse")
        self.model.summary()

    def train_pdm_model(self, epochs=100, steps_per_epoch=100, validation_steps=50, verbose=1):
        """
        Train the generate model based on provided epochs and steps.
        This function will also automatically generate the plot for loss history.
        """

        tf.keras.backend.clear_session()
        self.history = self.model.fit(
            self.train_data, epochs, steps_per_epoch, self.val_data, validation_steps, verbose
        )
        self.vs_an.plot_training_results_history(self.history)


if __name__ == "__main__":

    # Model dataset building parameters
    hist_window = 48
    horizon = 1
    train_split = 4906636
    batch_size = 256
    buffer_size = 150

    # Training parameters
    epochs = 150
    steps = 100
    validation_steps = 50
    verbose = 1

    e_rul = EngineRUL(DS_001)
    e_rul.visualize_data()
    e_rul.generate_tf_data_for_model(hist_window, horizon, train_split)
    e_rul.generate_pdm_model(epochs, steps, validation_steps, verbose)
