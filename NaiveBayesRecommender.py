import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB


class NaiveBayesRecommender:
    def __init__(self, tag_embeddings: np.ndarray):
        self.tag_embeddings = tag_embeddings

    def recommend_for_user(self, user_id: int, user_artists: pd.DataFrame, n=10):
        all_artists = np.arange(self.tag_embeddings.shape[0])
        liked_artists = user_artists[user_artists['userID'] == user_id]['artistID'].values

        if len(liked_artists) == 0:
            print(f"No history for thast user")
            return []


        all_set = set(all_artists)
        liked_set = set(liked_artists)
        negative_candidates = np.array(list(all_set - liked_set))

        np.random.seed(42)
        neg_samples = np.random.choice(negative_candidates, size=len(liked_artists)*5, replace=False)

        X_pos = self.tag_embeddings[liked_artists]
        y_pos = np.ones(len(liked_artists))

        X_neg = self.tag_embeddings[neg_samples]
        y_neg = np.zeros(len(neg_samples))

        X_train = np.vstack([X_pos, X_neg])
        y_train = np.concatenate([y_pos, y_neg])

        model = MultinomialNB()
        model.fit(X_train, y_train)

        unseen_artists = negative_candidates
        X_unseen = self.tag_embeddings[unseen_artists]
        probs = model.predict_proba(X_unseen)[:, 1]
        # ng.argsort sorts the values, [::-1] turns into ascending order, and then we take from index 0 to n
        top_idx = np.argsort(probs)[::-1][:n]
        top_artists = unseen_artists[top_idx]
        top_scores = probs[top_idx]

        return pd.DataFrame({'artistID': top_artists, 'score': top_scores})
