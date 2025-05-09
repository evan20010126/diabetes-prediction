import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go





def visualize_data(data_path, output_dir="./outputs/visualizations"):
    """
    Visualize the dataset with common plots and save them to files.
    
    Args:
        data_path (str): Path to the dataset (CSV file).
        output_dir (str): Directory to save the plots.
    """
    
    # Use ggplot style for consistent visuals
    plt.style.use("ggplot")
    plt.rcParams['axes.facecolor'] = '#fafafa'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Display the distribution of the target variable
    trace = go.Pie(
        labels = ['healthy', 'diabetic'],               
        values = data['Outcome'].value_counts(),        
        textfont = dict(size=15),
        opacity = 0.8,
        marker = dict(
            colors = ['lightskyblue', 'gold'],          
            line = dict(color = '#000000', width = 1.5) 
        )
    )
    layout = dict(title = 'Distribution of Outcome variable')
    fig = go.Figure(data = [trace], layout = layout)
    dis_path = os.path.join(output_dir, "target_distribution.png")
    fig.write_image(dis_path)
    print(f"Saved static image to {dis_path}")
    

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
    pairplot = sns.pairplot(data, diag_kind="kde", corner=True, palette="Set2", hue="Outcome")
    pairplot.fig.suptitle("Pairplot of Numerical Features", y=1.02)
    pairplot_path = os.path.join(output_dir, "pairplot.png")
    pairplot.savefig(pairplot_path)
    plt.close()
    print(f"Saved pairplot to {pairplot_path}")

    # Boxplot for each feature
    for column in data.select_dtypes(include=["float64", "int64"]).columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=data[column], palette="Set2", hue=data["Outcome"])
        plt.title(f"Boxplot of {column}")
        boxplot_path = os.path.join(output_dir, f"boxplot_{column}.png")
        plt.savefig(boxplot_path)
        plt.close()
        print(f"Saved boxplot of {column} to {boxplot_path}")


        
        
    # Distribution of target variable
    if "target" in data.columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=data["target"], palette="Set2")
        plt.title("Distribution of Target Variable")
        target_dist_path = os.path.join(output_dir, "target_distribution.png")
        plt.savefig(target_dist_path)
        plt.close()
        print(f"Saved target distribution to {target_dist_path}")
        
    # plotting missing values
    plot_missing_value(data, 'Outcome')



def plot_missing_value(dataset, key, output_dir="./outputs/visualizations"):
    """
    Plot and save a bar chart of missing values for each feature in the dataset.
    
    Args:
        dataset (pd.DataFrame): data
        key (str): Any column name, just used to get dataset length.
        output_dir (str): Folder to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    fix_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    
    for col in fix_cols:
        dataset[col] = dataset[col].replace(0, np.nan)


    null_feat = pd.DataFrame(len(dataset) - dataset.isnull().sum(), columns=['Count'])
    percentage_null = pd.DataFrame(dataset.isnull().sum() / len(dataset) * 100, columns=['Count']).round(2)     

    trace = go.Bar(
        x=null_feat.index,
        y=null_feat['Count'],
        opacity=0.8,
        text=percentage_null['Count'],
        textposition='auto',
        marker=dict(
            color='#7EC0EE',
            line=dict(color='#000000', width=1.5)
        )
    )
    layout = dict(title="Missing Values (count & %)")
    fig = go.Figure(data=[trace], layout=layout)

    mis_path = os.path.join(output_dir, "missing_values.png")
    fig.write_image(mis_path)
    print(f"Saved missing values PNG to: {mis_path}")
    
    
    
if __name__ == "__main__":
    # Example usage
    dataset_path = "./data/Pima_Indians_diabetes.csv"  # Replace with your dataset path
    visualize_data(dataset_path)

