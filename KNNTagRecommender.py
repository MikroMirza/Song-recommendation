import pandas as pd
import numpy as np

class KNNTagRecommender:
    def __init__(self, k: int = 5, tag_weight: float = 0.3):
        self.k = k
        self.tag_weight = tag_weight
        self.user_artist_matrix = None
        self.user_tag_matrix = None
        self.artist_tag_matrix = None

    def fit(self, user_artist_matrix: pd.DataFrame, user_tags: pd.DataFrame):
        self.user_artist_matrix = user_artist_matrix

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
        #Base KNN score
        neighbors = self._nearest_users(user_id)
        ids, sims = zip(*neighbors)
        neighbor_matrix = self.user_artist_matrix.loc[list(ids)]
        weighted = np.dot(sims, neighbor_matrix.values) / np.sum(sims)
        base_scores = pd.Series(weighted, index=self.user_artist_matrix.columns)

        #Tag preference
        user_tag_pref = self.user_tag_matrix.loc[user_id].values
        tag_scores = np.dot(user_tag_pref, self.artist_tag_matrix.values.T)
        tag_scores = pd.Series(tag_scores, index=self.artist_tag_matrix.index)

        #Combine score
        tag_scores_aligned = tag_scores.reindex(base_scores.index, fill_value=0)
        combined = base_scores * (1 - self.tag_weight) + tag_scores_aligned * self.tag_weight

        #Remove already listened artists
        user_vector = self.user_artist_matrix.loc[user_id]
        combined = combined[user_vector == 0]

        return combined.sort_values(ascending=False).head(n)