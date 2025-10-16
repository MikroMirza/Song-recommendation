import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from neuMF_Hybrid import NeuMF_Hybrid
from Preprocess import Preprocessor
from InteractionDataset import InteractionDataset
from KNNTagRecommender import KNNTagRecommender
from NaiveBayesRecommender import NaiveBayesRecommender

def recommend_neumf(model, user_id, num_items, top_n=10, device="cpu"):
    model.eval()
    with torch.no_grad():
        item_ids = torch.arange(num_items, device=device)
        user_ids = torch.full((num_items,), user_id, device=device)
        scores = model(user_ids, item_ids)
        top_scores, top_idx = torch.topk(scores, top_n)
        return item_ids[top_idx].cpu().numpy(), top_scores.cpu().numpy()

class CLI():
    def __init__(self) -> None:
        pp = Preprocessor(data_path="data")
        pp.run()

        #knn
        knn = KNNTagRecommender()
        knn.fit(pp.user_item_matrix, pp.user_tagged, pp)
        self.knn = knn

        #NaiveBayes
        tag_embeddings_np = np.load("processed/tag_embeddings.npy")
        tag_embeddings_torch = torch.tensor(tag_embeddings_np, dtype=torch.float32)
        self.nb = NaiveBayesRecommender(tag_embeddings_np)

        #hybrid
        num_users = int(pp.user_artists['userID'].max()) + 1
        num_items = int(pp.user_artists['artistID'].max()) + 1

        dataset = InteractionDataset(pp.user_artists, num_items)
        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        train_ds, val_ds = random_split(dataset, [train_len, val_len])

        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256)

        ckpt_path = "checkpoints/neumf_model.ckpt"
        os.makedirs("checkpoints", exist_ok=True)

        if os.path.exists(ckpt_path):
            model = NeuMF_Hybrid.NeuMF_Hybrid.load_from_checkpoint(ckpt_path)
        else:
            model = NeuMF_Hybrid.NeuMF_Hybrid(
                num_users=num_users,
                num_items=num_items,
                pretrained_tag_embeddings=tag_embeddings_torch
            )
            trainer = pl.Trainer(
                max_epochs=5,
                accelerator="auto",
                devices="auto",
                log_every_n_steps=10
            )
            trainer.fit(model, train_loader, val_loader)
            trainer.save_checkpoint(ckpt_path)

        self.model = model
        self.pp = pp
        self._user = -1
        self.top_n = 10

    def use_knn(self) -> None:
        rec = self.knn.recommend(self._user, self.top_n)
        print(rec[["artistID", "name", "score"]])
    
    def use_bayes(self) -> None:
        rec = self.nb.recommend_for_user(self._user, self.pp.user_artists, self.top_n)
        print(rec)

    def use_hybrid(self) -> None:
        pp = self.pp
        model = self.model

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device_str)

        num_items = int(pp.user_artists['artistID'].max()) + 1

        top_items, top_scores = recommend_neumf(model, user_id=self._user, num_items=num_items, top_n=self.top_n, device=device_str)

        rec_df_neumf = pd.DataFrame({'artistID': top_items, 'neuMF_score': top_scores})

        pp.artists["id"] = pd.to_numeric(pp.artists["id"], errors="coerce")
        pp.artists.dropna(subset=["id"], inplace=True)
        pp.artists["id"] = pp.artists["id"].astype(int)

        rec_df_neumf = rec_df_neumf.merge(pp.artists, left_on="artistID", right_on="id", how="left")
        print(rec_df_neumf[["artistID", "name", "neuMF_score"]])

    def choose_user(self) -> None:
        user_ids = self.pp.user_artists['userID'].unique()

        while True:
            try:
                text = input("Enter a user ID to generate recommendations for (or 'exit' to exit the program): ")
                if(text.upper() == "EXIT"):
                    break

                user_id = int(text)
                if user_id in user_ids:
                    self._user = user_id
                    print(f"User {user_id} selected.\n")
                    self.choose_method()
                else:
                    print("That user does not exist. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a numeric user ID.")

    def choose_top_n(self) -> None:
        while True:
            try:
                choice = int(input("Enter the number of recommendations you want: ").strip())
                self.top_n = choice
                break

            except ValueError:
                print("Invalid number, try again.")

    def choose_method(self) -> None:
        while True:
            try:
                print("Pick the algorithm you want to use")
                print("1. KNN")
                print("2. NaiveBayes")
                print("3. NCF and content-based embedding hybrid")
                print("4. Choose different user")
                choice = int(input(""))

                if choice == 1:
                    self.choose_top_n()
                    self.use_knn()
                elif choice == 2:
                    self.choose_top_n()
                    self.use_bayes()
                elif choice == 3:
                    self.choose_top_n()
                    self.use_hybrid()
                elif choice == 4:
                    break
                else:
                    print("Invalid choice")

            except ValueError:
                print("Invalid input, please try again.")

    def run(self) -> None:
        self.choose_user()