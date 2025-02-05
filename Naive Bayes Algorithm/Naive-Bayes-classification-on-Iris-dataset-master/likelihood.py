import matplotlib.pyplot as plt
import pandas as pd

# Create a synthetic likelihood table for demonstration
data = {
    "Petal Length (cm)": [1.5, 4.0, 6.5],
    "P(Setosa)": [0.8, 0.0, 0.0],
    "P(Versicolor)": [0.1, 0.7, 0.1],
    "P(Virginica)": [0.1, 0.3, 0.9],
}

likelihood_df = pd.DataFrame(data)

# Visualization of Likelihood Table
def visualize_likelihood_table(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.title("Likelihood Table Visualization", fontsize=14)
    plt.show()

# Call the visualization function
visualize_likelihood_table(likelihood_df)
