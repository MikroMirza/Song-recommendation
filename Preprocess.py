import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
from tqdm import tqdm
from data_load import DataLoader

tqdm.pandas()


class Preprocessor:
    def __init__(self, data_path: str, min_interactions_per_artist: int = 5):
        self.loader = DataLoader(data_path)
        self.min_interactions_per_artist = min_interactions_per_artist
        self.scaler = MinMaxScaler()
        self.stemmer = SnowballStemmer("english")

    def load_data(self):
        self.artists = self.loader.load_artists("artists.dat")
        self.tags = self.loader.load_tags("tags.dat")
        self.user_artists = self.loader.load_user_artists("user_artists.dat")
        self.user_tagged = self.loader.load_user_tagged_artist("user_taggedartists.dat")
        self.user_friends = self.loader.load_user_friends("user_friends.dat")

    def clean_and_filter(self):
        self.user_artists.drop_duplicates(inplace=True)
        self.user_tagged.drop_duplicates(inplace=True)

        # Filter artists with few interactions
        artist_counts = self.user_artists['artistID'].value_counts()
        valid_artists = artist_counts[artist_counts >= self.min_interactions_per_artist].index
        self.user_artists = self.user_artists[self.user_artists['artistID'].isin(valid_artists)]

    def normalize_weights(self):
        # Since the table user_artists is filled with USER_ID, ARTIST_ID and WEIGHT, weight represents amount of time a user listened to the artist
        # So if we have a user that listened to an artist 1 bajilion times, we do our best to scale it in a [0, 1] scope
        # Formula is (weight_of_user_artist - min_interactions)/(max_weight-min_interactions)
        self.user_artists['weight'] = self.scaler.fit_transform(
            self.user_artists[['weight']]
        )

    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        train, test = train_test_split(self.user_artists, test_size=test_size, random_state=random_state)
        train, val = train_test_split(train, test_size=val_size, random_state=random_state)
        self.train_df, self.val_df, self.test_df = train, val, test

    def create_user_item_matrix(self):
        self.user_item_matrix = self.user_artists.pivot_table(
            index='userID', columns='artistID', values='weight', fill_value=0
        )

    def stem_and_vectorize_tags(self):

        # Merge tag text with user_tagged info
        merged = self.user_tagged.merge(self.tags, on="tagID")
        # If artists has multiple tags given by multiple different user, this creates a massive fat string with all the tags.
        merged['tagValue'] = merged['tagValue'].progress_apply(
            lambda x: " ".join(self.stemmer.stem(word) for word in str(x).split())
        )

        # Group tags per artist
        artist_tags = merged.groupby('artistID')['tagValue'].apply(lambda x: " ".join(x)).reset_index()

        # Vectorize with built in TF-IDF (MAYBE WE CREATE OUR OWN TF-IDF DOWN THE ROUTE IDK)
        vectorizer = TfidfVectorizer(max_features=128)
        # Creates a vector that converts [metal, rock, metal, metal, metal, pop, metal, heavy metal, ...] into a vector of [0.73, 0.43, 0.11, ...]
        # So we transfer the tag names into the frequency of them being used for this artist vs for the rest of the artists
        tag_vectors = vectorizer.fit_transform(artist_tags['tagValue']).toarray()

        # Create embedding matrix aligned with artist IDs
        max_artist_id = self.user_artists['artistID'].max() + 1
        embedding_matrix = np.zeros((max_artist_id, tag_vectors.shape[1]))
        # Now we create a big massive matrix
        # Each ROW is an ID of a given ARTIST
        # Each COLUMN is a TAG for that ARTIST
        # The VALUSE in the matrix correspond to the TF-IDF weight for that TAG and ARTIST from the tag_vectors we created
        for idx, row in artist_tags.iterrows():
            artist_id = int(row['artistID'])
            embedding_matrix[artist_id] = tag_vectors[idx]

        self.tag_embeddings = embedding_matrix
        self.vectorizer = vectorizer

    def normalize_embeddings(self):
        # We normalize the tag_embeddings from Stemming just in case some fall out of the [0.00 , 1.00] scope
        self.tag_embeddings = MinMaxScaler().fit_transform(self.tag_embeddings)

    def save_processed_data(self, output_dir="processed"):
        os.makedirs(output_dir, exist_ok=True)
        self.train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        self.val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        self.test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

        np.save(os.path.join(output_dir, "tag_embeddings.npy"), self.tag_embeddings)
        self.user_item_matrix.to_csv(os.path.join(output_dir, "user_item_matrix.csv"))

    def run(self):
        self.load_data()
        self.clean_and_filter()
        self.normalize_weights()
        self.split_data()
        self.create_user_item_matrix()
        self.stem_and_vectorize_tags()
        self.normalize_embeddings()
        self.save_processed_data()
