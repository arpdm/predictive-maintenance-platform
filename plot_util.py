class PlotUtil:
    
    def __init__(self):
        print("Utility file for generic plotting and visualization")
    
    def generate_correlation_heatmap(self, data_frame):
        # Generate heatmap to figure out correlations between features and the RUL
        # This helps to determine which features to drop and which features to keep
        sns.heatmap(df_x_s_train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
        fig=plt.gcf()
        fig.set_size_inches(20,20)
        plt.show()