import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_data(data_path, output_dir="./outputs/visualizations"):
    """
    Visualize the dataset with common plots and save them to files.
    
    Args:
        data_path (str): Path to the dataset (CSV file).
        output_dir (str): Directory to save the plots.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    data = pd.read_csv(data_path)

    # Display basic information about the dataset
    print("Dataset Info:")
    print(data.info())
    print("\nDataset Description:")
    print(data.describe())

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved correlation heatmap to {heatmap_path}")

    # Pairplot for numerical features
    pairplot = sns.pairplot(data, diag_kind="kde", corner=True)
    pairplot.fig.suptitle("Pairplot of Numerical Features", y=1.02)
    pairplot_path = os.path.join(output_dir, "pairplot.png")
    pairplot.savefig(pairplot_path)
    plt.close()
    print(f"Saved pairplot to {pairplot_path}")

    # Boxplot for each feature
    for column in data.select_dtypes(include=["float64", "int64"]).columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=data[column])
        plt.title(f"Boxplot of {column}")
        boxplot_path = os.path.join(output_dir, f"boxplot_{column}.png")
        plt.savefig(boxplot_path)
        plt.close()
        print(f"Saved boxplot of {column} to {boxplot_path}")

    # Distribution of target variable (if exists)
    if "target" in data.columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=data["target"])
        plt.title("Distribution of Target Variable")
        target_dist_path = os.path.join(output_dir, "target_distribution.png")
        plt.savefig(target_dist_path)
        plt.close()
        print(f"Saved target distribution to {target_dist_path}")


if __name__ == "__main__":
    # Example usage
    dataset_path = "./data/Pima_Indians_diabetes.csv"  # Replace with your dataset path
    visualize_data(dataset_path)
