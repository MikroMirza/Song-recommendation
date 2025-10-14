import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class NeuMF_Hybrid(pl.LightningModule):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        gmf_embedding_dim: int = 64,
        mlp_embedding_dim: int = 64,
        tag_embedding_dim: int = 128,
        mlp_layers: list = [256, 128, 64],
        pretrained_tag_embeddings: torch.Tensor = None,
        lr: float = 0.001
    ):
        super().__init__()
        pass

    def forward(self, user_ids, item_ids):
        pass
    def training_step(self, batch, batch_idx):
        pass
    def validation_step(self, batch, batch_idx):
        pass
    def configure_optimizers(self):
        pass
