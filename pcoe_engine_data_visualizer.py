import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import gridspec

PLOT_LBL_SIZE = 17
COLOR_DIC_UNIT = {
    "Unit 1": "C0",
    "Unit 2": "C1",
    "Unit 3": "C2",
    "Unit 4": "C3",
    "Unit 5": "C4",
    "Unit 6": "C5",
    "Unit 7": "C6",
    "Unit 8": "C7",
    "Unit 9": "C8",
    "Unit 10": "C9",
    "Unit 11": "C10",
    "Unit 12": "C11",
    "Unit 13": "C12",
    "Unit 14": "C13",
    "Unit 15": "C14",
    "Unit 16": "C15",
    "Unit 17": "C16",
    "Unit 18": "C17",
    "Unit 19": "C18",
    "Unit 20": "C19",
}


class PcoeEngingeVis:
    def __init__(self, engine_data):
        print("Data analyzer and visualizer.")
        self.ed = engine_data

    def get_engine_units_in_dataset(self):
        print("Engine units in dataset: ", np.unique(self.ed.df_aux["unit"]))
        print("Engine units in test dataset: ", np.unique(self.ed.df_aux_test["unit"]))
        print("Engine units in train dataset: ", np.unique(self.ed.df_aux_train["unit"]))

    def plot_flight_classes(self):
        """
        Plot the engine units and their corresponding fligth class to
        findout the durations of each fligth based on the class
            Class 1 - Flight length (1-3) [h]
            Class 2 - Flight length (3-5) [h]
            Class 3 - Fligth length (5-7) [h]
        """

        plt.plot(self.ed.df_aux.unit, self.df_aux.Fc, "o")
        plt.tick_params(axis="x", labelsize=PLOT_LBL_SIZE)
        plt.tick_params(axis="y", labelsize=PLOT_LBL_SIZE)
        plt.xlabel("Unit # [-]", fontsize=PLOT_LBL_SIZE)
        plt.ylabel("Flight Class # [-]", fontsize=PLOT_LBL_SIZE)
        plt.show()
