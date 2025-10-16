import pandas as pd
import numpy as np
from Preprocess import Preprocessor

class KNNTagRecommender:
    def __init__(self, k: int = 5, tag_weight: float = 0.3):
        self.k = k
        self.tag_weight = tag_weight
        self.user_artist_matrix = None
        self.user_tag_matrix = None
        self.artist_tag_matrix = None

    def fit(self, user_artist_matrix: pd.DataFrame, user_tags: pd.DataFrame, pp: Preprocessor):
        self.user_artist_matrix = user_artist_matrix
        self._pp = pp

        #Build artist-tag matrix (1 or 0 - artist either has tag or he doesn't)
        artist_tags = user_tags[['artistID', 'tagID']].drop_duplicates()
        artist_tags['value'] = 1
        self.artist_tag_matrix = artist_tags.pivot_table(
            index='artistID',
            columns='tagID',
            values='value'
        ).fillna(0)

        #Build user-tag matrix (based on artists they listened to)
        common_artists = set(self.user_artist_matrix.columns) & set(self.artist_tag_matrix.index)
        ua = self.user_artist_matrix[list(common_artists)]
        at = self.artist_tag_matrix.loc[list(common_artists)]
        #user x tag = (user x artist) x (artist x tag)
        self.user_tag_matrix = np.dot(ua.values, at.values)
        self.user_tag_matrix = pd.DataFrame(
            self.user_tag_matrix,
            index=ua.index,
            columns=at.columns
        )

        print(f"Model trained with {len(ua)} users, {len(at)} artists, {len(at.columns)} tags.")

    #How we determine the similarity between users
    def _cosine_similarity(self, v1, v2):
        num = np.dot(v1, v2)
        den = np.linalg.norm(v1) * np.linalg.norm(v2)
        return num / den if den != 0 else 0

    def _nearest_users(self, user_id):
        target = self.user_artist_matrix.loc[user_id].values
        sims = {}
        for other in self.user_artist_matrix.index:
            if other == user_id:
                continue
            sims[other] = self._cosine_similarity(target, self.user_artist_matrix.loc[other].values)
        return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:self.k]

    def recommend(self, user_id, n=5):
        neighbors = self._nearest_users(user_id)
        ids, sims = zip(*neighbors)
        neighbor_matrix = self.user_artist_matrix.loc[list(ids)]
        weighted = np.dot(sims, neighbor_matrix.values) / np.sum(sims)
        base_scores = pd.Series(weighted, index=self.user_artist_matrix.columns)

        common_tags = self.user_tag_matrix.columns.intersection(self.artist_tag_matrix.columns)
        user_tag_pref = self.user_tag_matrix.loc[user_id, common_tags].values
        artist_tag_mat = self.artist_tag_matrix[common_tags]

        tag_scores = np.dot(user_tag_pref, artist_tag_mat.values.T)
        tag_scores = pd.Series(tag_scores, index=artist_tag_mat.index)

        tag_scores_aligned = tag_scores.reindex(base_scores.index, fill_value=0)
        combined = base_scores * (1 - self.tag_weight) + tag_scores_aligned * self.tag_weight

        user_vector = self.user_artist_matrix.loc[user_id]
        combined = combined[user_vector == 0]
        combined = combined.sort_values(ascending=False).head(n)

        combined_df = combined.to_frame(name='score').reset_index()
        combined_df['artistID'] = combined_df['artistID'].astype(str)
        combined_df = combined_df.merge(self._pp.artists, left_on='artistID', right_on='id', how='left')

        return combined_df