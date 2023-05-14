import pandas as pd
from matplotlib import pyplot as plt


# function that takes the path to a csv file. Reads the file and creates a plot
# with epochs on the x-axis and valid_loss on the y-axis
def plot_loss(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['valid_loss'])
    plt.plot(df['epoch'], df['valid_loss'])
    plt.xlabel('epoch')
    plt.ylabel('valid_loss')
    plt.title('valid_loss vs epoch')
    plt.show()


plot_loss("lightning_logs/version_4/metrics.csv")
# df = pd.read_csv("lightning_logs/version_4/metrics.csv")