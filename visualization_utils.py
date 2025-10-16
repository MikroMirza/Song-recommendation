# visualization_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ======================================================
# 📊 USER–ARTIST INTERACTION DISTRIBUTIONS
# ======================================================
def plot_user_interaction_distribution(user_artists_df: pd.DataFrame):
    interaction_counts = user_artists_df['userID'].value_counts()
    plt.figure(figsize=(10, 5))
    sns.histplot(interaction_counts, bins=50, kde=True)
    plt.title("📊 Number of Interactions per User")
    plt.xlabel("Interactions")
    plt.ylabel("Number of Users")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_artist_popularity_distribution(user_artists_df: pd.DataFrame):
    artist_counts = user_artists_df['artistID'].value_counts()
    plt.figure(figsize=(10, 5))
    sns.histplot(artist_counts, bins=50, kde=True, color='orange')
    plt.title("🎤 Artist Popularity Distribution")
    plt.xlabel("Number of Users")
    plt.ylabel("Number of Artists")
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_tag_embeddings_pca(tag_embeddings: np.ndarray, n_points: int = 500):
    if tag_embeddings.shape[0] > n_points:
        tag_embeddings = tag_embeddings[:n_points]
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(tag_embeddings)
    plt.figure(figsize=(7, 7))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.6)
    plt.title(f"🧭 Tag Embeddings PCA projection ({n_points} points)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_label_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    counts = [len(train_df), len(val_df), len(test_df)]
    labels = ["Train", "Validation", "Test"]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=counts, palette="Blues_r")
    plt.title("📊 Dataset Split Sizes")
    plt.ylabel("Number of Interactions")
    plt.show()


def plot_evaluation_comparison(metrics_dict):
    df = pd.DataFrame(metrics_dict).T
    df.plot(kind='bar', figsize=(10,6))
    plt.title("📈 Evaluation Metrics Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(title="Metric")
    plt.grid(axis='y', alpha=0.3)
    plt.show()
