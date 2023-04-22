import torch
from torch import nn


class SingleModalityEmbedding(nn.Module):
    def __init__(self,
                 args,
                 feature_embed_dim,
                 item_num):
        super().__init__()
        self.args = args
        self.feature_embed_dim = feature_embed_dim
        self.item_num = item_num

        self.embeddings = nn.Embedding(item_num + 3, self.feature_embed_dim, padding_idx=0)
        self.mask = nn.Embedding(item_num + 3, 1, padding_idx=0)

    def weight_init(self, embedding, mask):
        embedding = embedding.reshape(-1, self.feature_embed_dim)
        embedding = torch.cat([torch.zeros((3, self.feature_embed_dim), device=self.args.device), embedding], dim=0)

        self.embeddings.weight.data.copy_(embedding)
        self.embeddings.weight.requires_grad = False

        mask = mask.reshape(-1, 1).type(torch.float)
        mask = torch.cat([torch.tensor([[0], [1], [1]], device=self.args.device), mask], dim=0)

        self.mask.weight.data.copy_(mask)
        self.mask.weight.requires_grad = False

    def forward(self, input_ids):
        embedding = self.embeddings(input_ids)
        mask = self.mask(input_ids)
        mask = mask.squeeze(-1).type(torch.bool)
        return embedding.detach(), mask.detach()


class ModalityEmbedding(nn.Module):
    def __init__(self,
                 args,
                 **kwargs):
        super().__init__()
        self.args = args

        assert args.train_item_num == args.eval_item_num
        self.item_num = args.train_item_num

        self.vision_embeddings = SingleModalityEmbedding(args, args.vision_feature_embed_dim, self.item_num)
        self.text_embeddings = SingleModalityEmbedding(args, args.text_feature_embed_dim, self.item_num)

    def weight_init(self, vision, vision_mask, text, text_mask):
        self.vision_embeddings.weight_init(vision, vision_mask)
        self.text_embeddings.weight_init(text, text_mask)

    def forward(self, input_ids):
        vision, vision_mask = self.vision_embeddings(input_ids)
        text, text_mask = self.text_embeddings(input_ids)
        return vision, vision_mask, text, text_mask


class FusionInputTransform(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.vision_feature_embed_dim = args.vision_feature_embed_dim
        self.text_feature_embed_dim = args.text_feature_embed_dim
        self.fusion_embed_dim = args.fusion_embed_dim

        self.vision_transform = nn.Linear(self.vision_feature_embed_dim, self.fusion_embed_dim)
        self.text_transform = nn.Linear(self.text_feature_embed_dim, self.fusion_embed_dim)

        self.vision_layer_norm = nn.LayerNorm(self.fusion_embed_dim)
        self.text_layer_norm = nn.LayerNorm(self.fusion_embed_dim)

    def forward(self, vision, vision_mask, text, text_mask):
        vision_hidden = self.vision_layer_norm(self.vision_transform(vision))
        text_hidden = self.text_layer_norm(self.text_transform(text))

        hidden = torch.stack([vision_hidden, text_hidden], dim=2)     # [N, L, M, D]
        mask = torch.stack([vision_mask, text_mask], dim=2)     # [N, L, M]
        return hidden, mask


class FusionEmbedding(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.fusion_embed_dim = args.fusion_embed_dim

        self.position_embeddings = nn.Embedding(args.max_seq_length, self.fusion_embed_dim)
        self.modal_type_embeddings = nn.Embedding(2, self.fusion_embed_dim)
        self.mask_embeddings = nn.Embedding(1, self.fusion_embed_dim)
        self.prompt_embeddings = nn.Embedding(1, self.fusion_embed_dim)

        self.layer_norm = nn.LayerNorm(self.fusion_embed_dim)
        self.dropout = nn.Dropout(args.fusion_embed_dropout)

    def forward(self, input_ids, hidden):
        N, L, M, D = hidden.shape

        mask_token = self.mask_embeddings(torch.tensor(0, device=self.args.device))
        hidden = torch.where(input_ids.reshape(N, L, 1, 1).expand_as(hidden) == 1, mask_token, hidden)
        prompt_token = self.prompt_embeddings(torch.tensor(0, device=self.args.device))
        hidden = torch.where(input_ids.reshape(N, L, 1, 1).expand_as(hidden) == 2, prompt_token, hidden)

        position_ids = torch.arange(L, dtype=torch.int, device=self.args.device)
        position_ids = position_ids.unsqueeze(-1).repeat(N, 1, M)
        type_ids = torch.arange(2, dtype=torch.int, device=self.args.device)

        hidden = hidden + self.position_embeddings(position_ids) + self.modal_type_embeddings(type_ids)
        hidden = self.layer_norm(hidden)
        hidden = self.dropout(hidden)
        return hidden   # [N, L, M, D]


class FusionOutputTransform(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.fusion_embed_dim = args.fusion_embed_dim

        self.vision_transform = nn.Linear(self.fusion_embed_dim, self.fusion_embed_dim)
        self.text_transform = nn.Linear(self.fusion_embed_dim, self.fusion_embed_dim)

        self.vision_layer_norm = nn.LayerNorm(self.fusion_embed_dim)
        self.text_layer_norm = nn.LayerNorm(self.fusion_embed_dim)

    def forward(self, hidden_states, mask):
        vision_hidden = self.vision_layer_norm(self.vision_transform(hidden_states[:, :, 0, :]))
        text_hidden = self.text_layer_norm(self.text_transform(hidden_states[:, :, 1, :]))

        fusion_hidden = torch.stack([vision_hidden, text_hidden], dim=-2)
        fusion_hidden = torch.where(mask.unsqueeze(-1).expand_as(fusion_hidden), fusion_hidden, float(-1e+4))
        hidden = torch.max(fusion_hidden, dim=-2).values

        return hidden   # [N, L, D]




















