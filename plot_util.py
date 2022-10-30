import seaborn as sns
import matplotlib.pyplot as plt


class PlotUtil:
    def __init__(self):
        print("Utility file for generic plotting and visualization")

    def generate_correlation_heatmap(self, data_frame):
        """
        Generate heatmap to figure out correlations between features and labels
        This helps to determine which features to drop and which features to keep
        """

        sns.heatmap(data_frame.corr(), annot=True, cmap="RdYlGn", linewidths=0.2)
        fig = plt.gcf()
        fig.set_size_inches(20, 20)
        plt.show()

    def generate_training_history_accuracy_and_loss_plot(self, history, image_location=None):
        """
        Base on model fitting history plots the accuracy of the model per epoch for both training and validation sets.
        Base on model fitting history plots the loss of the model per epoch for both training and validation sets.
        If user provides image location, store the figure as png
        """

        # Accuracy sub-plot
        fig_acc = plt.figure(figsize=(20, 5), dpi=300)
        plt.subplot(1, 2, 1)
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.plot(history.history["accuracy"], label="training")
        plt.plot(history.history["val_accuracy"], label="validation")
        plt.legend(["training", "validation"], loc="upper left")

        # Loss sub-plot
        plt.subplot(1, 2, 2)
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.plot(history.history["loss"], label="training")
        plt.plot(history.history["val_loss"], label="validation")
        plt.legend(["training", "validation"], loc="upper right")

        if image_location is not None:
            fig_acc.savefig(f"{image_location}/model_accuracy_loss.png")

    def generate_predicted_vs_true_plot(self, y_true, y_pred, y_label):
        """
        Plots both predicted vs true model outputs in the same figure.
        Y_label : descriptive text for y-axis on the plot
        """

        plt.figure(figsize=(16, 9))
        plt.plot(list(y_true.reshape(-1)))
        plt.plot(list(y_pred.reshape(-1)))
        plt.title("Actual vs Predicted")
        plt.ylabel(y_label)
        plt.legend(("Actual", "predicted"))
        plt.show()

    def generate_heat_map(self, data_frame):
        plt.figure(figsize=(10, 7))
        sns.heatmap(data_frame, annot=True)

    def generate_about_to_fail_graphs(self, y_true, y_pred):
        """
        Generates two sub-figures. One for the ground truth, one for predicted values.
        """

        plt.figure(figsize=(50, 10))
        plt.subplot(2, 1, 1)
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.title("About to Fail (ground truth)")
        plt.plot(y_true, "r")
        plt.subplot(2, 1, 2)
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.title("About to Fail (Predicted)")
        plt.plot(y_pred, "b")
