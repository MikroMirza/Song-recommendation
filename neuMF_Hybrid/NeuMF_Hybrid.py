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
        self.save_hyperparameters()
        self.user_embedding_gmf = nn.Embedding(num_users,gmf_embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, gmf_embedding_dim)

        self.user_embedding_mlp = nn.Embedding(num_users,mlp_embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, mlp_embedding_dim)

        if pretrained_tag_embeddings is not None:
            self.tag_embeddings = nn.Embedding.from_pretrained(pretrained_tag_embeddings, freeze=False)
        else:
            self.tag_embeddings = nn.Embedding(num_items, tag_embedding_dim)

        mlp_input_dim = mlp_embedding_dim*2+tag_embedding_dim
        layers=[]
        for dim in mlp_layers:
            layers.append(nn.Linear(mlp_input_dim,dim))
            layers.append(nn.ReLU())
            mlp_input_dim = dim
        self.mlp = nn.Sequential(*layers)

        final_dim = gmf_embedding_dim+mlp_layers[-1]
        self.fc = nn.Linear(final_dim,1)
        self.lr = lr

    def forward(self, user_ids, item_ids):
        gmf_user = self.user_embedding_gmf(user_ids)
        gmf_index = self.item_embedding_gmf(item_ids)
        gmf_output = gmf_user*gmf_index

        mlp_user = self.user_embedding_mlp(user_ids)
        mlp_index = self.item_embedding_mlp(item_ids)
        tag_index = self.tag_embeddings(item_ids)

        mlp_input = torch.cat([mlp_user,mlp_index,tag_index],dim=-1)
        mlp_output = self.mlp(mlp_input)

        x = torch.cat([gmf_output,mlp_output],dim=-1)
        logits=self.fc(x)
        return torch.sigmoid(logits).squeeze()

    def training_step(self, batch, batch_idx):
        users, items, labels = batch
        preds = self.forward(users,items)
        loss = F.binary_cross_entropy(preds, labels.float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        users, items, labels = batch
        preds = self.forward(users, items)
        loss = F.binary_cross_entropy(preds, labels.float())
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
