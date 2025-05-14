from utils.logger_util import CustomLogger as logger
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np


def save_eigenvector_matrix(eigenvector_matrix,
                            feature_names_before_PCA,
                            out_root,
                            title="Eigenvector_Matrix"):
    """
    Save the eigenvector matrix as a CSV file.
    Args:
        eigenvector_matrix (np.ndarray): The eigenvector matrix from PCA. # .components_.T shape: (n_features, n_components)
        feature_names_before_PCA (list): The names of the features before PCA.
        out_root (str, optional): The root directory to save the plots. Defaults to None.
    """
    logger.info("Saving eigenvector matrix...")

    eigenvector_matrix_df = pd.DataFrame(
        eigenvector_matrix,
        columns=[f"PC{i+1}" for i in range(eigenvector_matrix.shape[1])],
        index=feature_names_before_PCA)

    # Save eigenvector matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(eigenvector_matrix_df, annot=True, cmap='coolwarm', center=0)
    plt.title(title)
    plt.xlabel("Principal Components")
    plt.ylabel("Original Features")
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, f"{title}.png"))


def save_importance(importance,
                    feature_names,
                    out_root,
                    title="Feature_Importance"):

    logger.info(f"Saving {title}...")
    plt.figure(figsize=(10, 15))
    plt.bar(range(len(importance)), importance, alpha=0.5, align='center')
    plt.ylabel("Importance")
    plt.xlabel("Features")
    plt.xticks(range(len(importance)), feature_names, rotation=90)
    plt.title(title)
    plt.savefig(os.path.join(out_root, f"{title}.png"))


def compute_importance(eigenvector_matrix):
    """
    Compute and save the importance of features.
    """

    importance = np.abs(eigenvector_matrix).sum(axis=1)
    importance = importance / importance.sum()

    return importance


def compute_eigenvector_matrix_with_weighted(eigenvector_matrix, weighted):
    """
    Compute the eigenvector matrix with weighted features.
    """

    logger.info("Computing eigenvector matrix with weighted features...")
    weighted_eigenvector_matrix = eigenvector_matrix * weighted[np.newaxis, :]

    return weighted_eigenvector_matrix


if __name__ == "__main__":
    # Example usage
    """
    pca = PCA(n_components=pca_cfg['n_components'], random_state=SEED)
    eigenvector_matrix = pca.components_.T  # shape: (n_features, n_components)
    importance = compute_importance(eigenvector_matrix=eigenvector_matrix)
    save_eigenvector_matrix(eigenvector_matrix=eigenvector_matrix,
                            feature_names_before_PCA=df_train.columns[:-1],
                            out_root=args.save.out_root,
                            title="PCA_Eigenvector_Matrix")
    save_importance(importance=importance,
                    feature_names=df_train.columns[:-1],
                    out_root=args.save.out_root,
                    title="PCA_Feature_Importance")
    """
    """
    weighted_eigenvector_matrix = compute_eigenvector_matrix_with_weighted(
    eigenvector_matrix, clf.feature_importances_)
    importance = compute_importance(
    eigenvector_matrix=weighted_eigenvector_matrix)
    save_importance(importance=importance,
                feature_names=df_train.columns[:-1],
                out_root=args.save.out_root,
                title="Weighted_PCA_Feature_Importance")
    # Save eigenvector matrix
    save_eigenvector_matrix(
    eigenvector_matrix=weighted_eigenvector_matrix,
    feature_names_before_PCA=df_train.columns[:-1],
    out_root=args.save.out_root,
    title="Weighted_PCA_Eigenvector_Matrix")
    """
    pass
