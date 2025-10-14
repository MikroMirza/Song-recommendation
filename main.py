from Preprocess import Preprocessor
from KNNTagRecommender import KNNTagRecommender

if __name__ == "__main__":
    pp = Preprocessor(data_path="data")
    pp.run()

    knnRecommender = KNNTagRecommender(5, 0.3)
    knnRecommender.fit(pp.user_item_matrix, pp.user_tagged)
    print(knnRecommender.recommend(2))