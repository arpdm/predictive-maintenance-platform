from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from pandas import DataFrame

import pickle
import keras
import tensorflow as tf
import numpy as np
import pandas as pd

OPTIMIZER = "adam"
DENSE_ACTIVATION = "sigmoid"
LOSS = "binary_crossentropy"


class Mark001Model:
    def __init__(self):
        print("Different PdM model collection to be tested.")
        self.model = None

    def create_binary_classifier_model(self, window, features, units_first_layer, units_second_layer, dropout_rate):
        """
        LSTM has long-term memory, which is needed for predicting anomalies in the times-series
        Dropout:
            Dropout is added because it helps reduce overfilling. It essentially drops neurons randomly.
            This in turn helps generalization of the model.
        Final Layer:
            The last layer is a sigmoid function so it can determine weather unit will fail withing time horizon or not.
        Activation Function:
            The sigmoid is applied to a dense layer so it gets applied to every neuron. This layer is a fully connected layer.
        Loss Function:
            The loss function used in this model is binary_crossentropy since we only have two classes of 1 and 0s.
        Optimizer:
            The optimizer essentially defines how to adjust neuron weights in response to inaccurate predictions.
            In this case, we use Adam optimizer.
            Adam is used since it learns fast and stable over wide range of learning rates and requires relatively low memory.
            The default learning rate used by Keras is 0.001.
        """
        mark_001 = Sequential()
        mark_001.add(LSTM(input_shape=(window, features), units=units_first_layer, return_sequences=True))
        mark_001.add(Dropout(dropout_rate))
        mark_001.add(LSTM(units=units_second_layer, return_sequences=False))
        mark_001.add(Dropout(dropout_rate))
        mark_001.add(Dense(units=1, activation=DENSE_ACTIVATION))
        mark_001.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])
        mark_001.summary()
        self.model = mark_001
        return mark_001

    def fit_model(self, x_train, y_train, epochs=40, steps=100, batch_size=10, patience=10):
        """
        Trains the model based on provided x and y training data frames.
        Epoch : by default is 40. However, it can be adjusted to increase accuracy/
        Steps: Steps per epoch by default is 100. This is also adjustable by the caller.
        Batch_Size : by default set to 10. This indicates number of samples used per iteration.
        Patience :  This parameter is used for early stopping.
                    Essentially if no improvement is made per number of iteration/epoch specified by this parameter,
                    model will stop training.
        """
        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            steps_per_epoch=steps,
            batch_size=batch_size,
            validation_split=0.05,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=patience, verbose=0, mode="min")
            ],
        )

        return history

    def save_model(self, history, path, model_name):
        """
        Saves the model to provided path for later use with the trained weights.
        The model itself is parsed to json file and saved to the path.
        The trained weights are saved to h5 file in the same path provided.
        """
        # Dump training history to file
        filename = open(f"{path}/history", "wb")
        pickle.dump(history, filename)
        filename.close()

        # serialize model to JSON
        model_json = model.to_json()
        with open(f"{path}/{model_name}.json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(f"{path}/{model_name}.h5")
        print("Saved model to disk")

    def load_model(self, model_path):
        """
        Loads existing model based on path provided.
        First loads the json file and gets the model from it.
        The loads the trained weights from saved h5 file to the model.
        The model is the compiled using LOSS function and OPTIMIZER
        """
        json_file = open(f"{model_path}.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

        # load weights into new model
        self.model.load_weights(f"{model_path}.h5")
        self.model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])

    def evaluate_model(self, x_train, y_train, batch_size):
        scores = self.model.evaluate(x_train, y_train, verbose=1, batch_size=batch_size)
        print("Accuracy of predictions made on data used for training:\n{}".format(scores[1]))

    def test_model(self, x_test, y_test):
        """
        This function tests the model by making predictions using s and y test datasets.
        It then performs some analysis by comparing true and predicted values.

        Outputs: Predicted Y values, True Y values, confusion matrix
        """
        # Since tensorflow down not have binary prediction anymore, we are going to put a checkpoint.
        # If the value is above 0.5, set it as 1 otherwise 0.
        y_pred = (self.model.predict(x_test) > 0.5).astype("int32")
        y_true = y_test

        # Prediction analysis
        recall = recall_score(y_true, y_pred)
        test_mae_loss = np.mean(np.abs(y_pred.reshape(1, -1) - y_true.reshape(1, -1)), axis=1)
        cm = pd.DataFrame(confusion_matrix(y_true, y_pred))

        # Test prediction report
        print("\nAccuracy of model on test data = " + str(accuracy_score(y_true, y_pred)))
        print("\nRecall = " + str(recall))
        print("\MAE Loss = " + str(test_mae_loss))

        return (y_pred, y_true, cm)

    def predict(self, x):
        """
        Makes predictions that are not part of training or test datasets.
        Outputs : Y Predicted
        """

        # Since tensorflow down not have binary prediction anymore, we are going to put a checkpoint.
        # If the value is above 0.5, set it as 1 otherwise 0.
        return (self.model.predict(x) > 0.5).astype("int32")
