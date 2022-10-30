from asyncio.windows_events import NULL
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
        """ """
        fig_acc = plt.figure(figsize=(20, 5), dpi=300)
        plt.subplot(1, 2, 1)
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.plot(history.history["accuracy"], label="training")
        plt.plot(history.history["val_accuracy"], label="validation")
        plt.legend(["training", "validation"], loc="upper left")

        plt.subplot(1, 2, 2)
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.plot(history.history["loss"], label="training")
        plt.plot(history.history["val_loss"], label="validation")
        plt.legend(["training", "validation"], loc="upper right")
        fig_acc.savefig(f"{image_location}/model_loss.png")
