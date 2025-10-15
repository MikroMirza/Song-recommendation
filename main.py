import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from neuMF_Hybrid import NeuMF_Hybrid
from Preprocess import Preprocessor
from InteractionDataset import InteractionDataset


def recommend_neumf(model, user_id, num_items, top_n=10, device="cpu"):
    model.eval()
    with torch.no_grad():
        item_ids = torch.arange(num_items, device=device)
        user_ids = torch.full((num_items,), user_id, device=device)
        scores = model(user_ids, item_ids)
        top_scores, top_idx = torch.topk(scores, top_n)
        return item_ids[top_idx].cpu().numpy(), top_scores.cpu().numpy()


if __name__ == "__main__":
    pp = Preprocessor(data_path="data")
    pp.run()

    tag_embeddings_np = np.load("processed/tag_embeddings.npy")
    tag_embeddings_torch = torch.tensor(tag_embeddings_np, dtype=torch.float32)

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

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    top_items, top_scores = recommend_neumf(model, user_id=100, num_items=num_items, top_n=10, device=device_str)

    rec_df_neumf = pd.DataFrame({'artistID': top_items, 'neuMF_score': top_scores})

    pp.artists["id"] = pd.to_numeric(pp.artists["id"], errors="coerce")
    pp.artists.dropna(subset=["id"], inplace=True)
    pp.artists["id"] = pp.artists["id"].astype(int)

    rec_df_neumf = rec_df_neumf.merge(pp.artists, left_on="artistID", right_on="id", how="left")
    print(rec_df_neumf[["artistID", "name", "neuMF_score"]])
