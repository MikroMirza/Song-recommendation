import os
import pandas as pd

class DataLoader:
    def __init__(self, path:str):
        self.path = path

    def load_file(self, filename: str, columns: list[str]) -> pd.DataFrame:
        filepath = os.path.join(self.path, filename)
        encodings_to_try = ["utf-8", "utf-7", "latin1", "ISO-8859-1", "cp1252"]

        for enc in encodings_to_try:
            try:
                df = pd.read_csv(
                    filepath,
                    delimiter="\t",
                    header=None,
                    names=columns,
                    encoding=enc
                )
                print(f"[INFO] Loaded {filename} successfully with encoding {enc}")
                return df
            except UnicodeDecodeError:
                continue

        raise Exception(f" Could not load {filename}")

    def load_artists(self, filename: str) ->pd.DataFrame:
        # ARTIST:           id     name     url     pictureURL
        return self.load_file(filename=filename, columns=["id","name","url","pictureURL"])

    def load_tags(self,filename:str)->pd.DataFrame:
        # TAGS:             tagID    tagValue
        return self.load_file(filename=filename, columns=["tagID", "tagValue"])

    def load_user_artists(self,filename:str) -> pd.DataFrame:
        # USER ARTISTS:     userID   artistID   weight
        return self.load_file(filename=filename, columns=["userID", "artistID", "weight"])

    def load_user_tagged_artist(self,filename:str)->pd.DataFrame:
        # User TaggedArtists: userID    artistID    tagID   day    month    year
        return self.load_file(filename=filename, columns=["userID", "artistID", "tagID", "day", "month", "year"])

    def load_user_tagged_artists_timestamps(self,filename:str)-> pd.DataFrame:
        # user tagged artists timestamps: userID    artistID    tagID   timestamp
        return self.load_file(filename=filename, columns=["userID", "artistID", "tagID", "timestamp"])

    def load_user_friends(self, filename:str)->pd.DataFrame:
        # USER FRIENDS:
        return self.load_file(filename=filename, columns=["userID", "friendID"])



