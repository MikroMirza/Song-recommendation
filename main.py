import numpy as np
import pandas as pd

from NaiveBayesRecommender import NaiveBayesRecommender
from Preprocess import Preprocessor
from KNNTagRecommender import KNNTagRecommender

if __name__ == "__main__":
    pp = Preprocessor(data_path="data")
    pp.run()

    knnRecommender = KNNTagRecommender(5, 0.3)
    knnRecommender.fit(pp.user_item_matrix, pp.user_tagged)
    recommendations = knnRecommender.recommend(2)

    # rec_df = recommendations.reset_index()
    # rec_df.columns = ["artistID", "score"]
    # rec_df["artistID"] = rec_df["artistID"].astype(int)
    #
    # # Clean artist IDs
    # pp.artists["id"] = pd.to_numeric(pp.artists["id"], errors="coerce")
    # pp.artists = pp.artists.dropna(subset=["id"])
    # pp.artists["id"] = pp.artists["id"].astype(int)
    #
    # rec_df = rec_df.merge(pp.artists, left_on="artistID", right_on="id", how="left")
    # print(rec_df[["artistID", "name", "score"]])
    tag_embeddings = np.load("processed/tag_embeddings.npy")

    nb_recommender = NaiveBayesRecommender(tag_embeddings)
    recommendations = nb_recommender.recommend_for_user(2, pp.user_artists, n=5)

    pp.artists["id"] = pd.to_numeric(pp.artists["id"], errors="coerce")
    pp.artists.dropna(subset=["id"], inplace=True)
    pp.artists["id"] = pp.artists["id"].astype(int)

    rec_df = recommendations.merge(pp.artists, left_on="artistID", right_on="id", how="left")
    print(rec_df[["artistID", "name", "score"]])
