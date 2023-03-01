# predictive-maintenance-platform

A platform for PdM. This repository has helper functions, classes to generalize the predictive maintenance platform.
At the moment, this is a prototype and it doesn't include the full A-Z pipelin.

## How to run the code

To run the latest working version, open the **experiments* colab notebook in google colab.

The colab file connects to Google Drive where the data are saved and the repository is cloned. The colab file grabs the classes from there.
So every time there are changes made to the repo, from terminal inside google colab, pull the chages to where the repo is cloned in google drive.

## Dataset Used

For development of the model we are using NASA's PCoE turbofan run to failure dataset.
Dataset description along with its variable can be found in the dataset paper written by 
Manuel Arias Chao,Chetan Kulkarni, Kai Goebel and Olga Fink. https://dx.doi.org/10.3390/data6010005

## Useful Code and References

https://gallery.azure.ai/Experiment/Predictive-Maintenance-Step-1-of-3-data-preparation-and-feature-engineering-2

https://github.com/Azure/lstms_for_predictive_maintenance 

https://github.com/Azure-Samples/MachineLearningSamples-DeepLearningforPredictiveMaintenance/blob/master/Code/2_model_building_and_evaluation.ipynb

https://github.com/mapr-demos/predictive-maintenance/blob/master/notebooks/jupyter/LSTM%20For%20Predictive%20Maintenance-ian01.ipynb

https://www.kaggle.com/datasets/behrad3d/nasa-cmaps 

https://www.kaggle.com/code/scratchpad/notebook5cc58ab447/edit 

