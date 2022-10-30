from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import EarlyStopping
from keras.models import model_from_json

import pickle
import keras
import tensorflow as tf

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
        Dropout is added because it helps reduce overfilling. It essentially drops neurons randomly.
        This in turn helps generalization of the model.
        The last layer is a sigmoid function so it can determine weather unit will fail withing time horizon or not.
        The sigmoid is applied to a dense layer so it gets applied to every neuron. This layer is a fully connected layer.
        The loss function used in this model is binary_crossentropy since we only have two classes of 1 and 0s.
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
        """ """
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

    def save_model(self, history, model, path, model_name):
        """ """
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
        """ """
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

    def predict(self, x_test, y_test, engine_id):
        print("test")