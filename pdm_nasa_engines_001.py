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

import os
import h5py
import time
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from pandas import DataFrame
from matplotlib import gridspec
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

# Params


class EngineRUL:
    def __init__(self):
        print("Engine RUL Predictor.")



if __name__ == "__main__":

    e_rul = EngineRUL()
    e_rul.load_hdf5_to_numpy_arr(DS_001)
    e_rul.get_engine_units_in_dataset()
    e_rul.plot_flight_classes()
    e_rul.show_engine_health_parameter_stats()
    e_rul.generate_engine_health_parameter_graphs()
    e_rul.generate_hpt_eff_over_cycles_all_engines()
    e_rul.generate_sensor_readings_graphs_single_unit(1)
    e_rul.generate_sensor_readings_graphs_single_unit_single_cycle(1, 2)
    e_rul.plot_health_states_for_all_engines()
    e_rul.generate_flight_profle_single_unit_single_cycle(1, 2)
    e_rul.generate_flight_envelope()
    e_rul.generate_kde_estimations_of_flight_profile()
