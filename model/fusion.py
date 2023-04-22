import os
import torch
from model.fusion_module import *
from model.transformer import Transformer


class FusionEncoder(nn.Module):
    def __init__(self, args, **kwargs):

        super().__init__()
        self.args = args

        self.input_transform = FusionInputTransform(args, **args)
        self.embedding = FusionEmbedding(args, **args)
        self.encoder_blocks = Transformer(args.fusion_embed_dim,
                                          args.fusion_layers,
                                          args.fusion_heads,
                                          args.fusion_feedforward_dim,
                                          args.fusion_dropout)
        self.output_transform = FusionOutputTransform(args, **args)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def save_pretrained(self, save_path):
        state_file = os.path.join(save_path, "fusion_model.pth")
        state = {
            "input": self.input_transform.state_dict(),
            "embedding": self.embedding.state_dict(),
            "blocks": self.encoder_blocks.state_dict(),
            "output": self.output_transform.state_dict()
        }
        torch.save(state, state_file)

    def from_pretrained(self, load_path, map_location):
        state_file = os.path.join(load_path, "fusion_model.pth")
        if not os.path.exists(state_file):
            return
        state = torch.load(state_file, map_location=map_location)
        del state["embedding"]["position_embeddings.weight"]
        self.input_transform.load_state_dict(state["input"], strict=False)
        self.embedding.load_state_dict(state["embedding"], strict=False)
        self.encoder_blocks.load_state_dict(state["blocks"], strict=False)
        self.output_transform.load_state_dict(state["output"], strict=False)

    def forward(self, input_ids, vision, vision_mask, text, text_mask, mode=None):
        embed, mask = self.input_transform(vision, vision_mask, text, text_mask)
        embed = self.embedding(input_ids, embed)
        inputs = torch.flatten(embed, 1, 2)
        hidden = self.encoder_blocks(inputs, ~mask.reshape(inputs.shape[0], -1))
        hidden = hidden.reshape(embed.shape)
        hidden = self.output_transform(hidden, mask)  # [N, L, D]
        mask = (input_ids >= 3)     # [N, L]

        return hidden, mask


class FusionEncoderForMaskedLM(FusionEncoder):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        self.ce_mask_ratio = args.ce_mask_ratio
        self.modality_embedding = ModalityEmbedding(args, **args)

    def modality_embedding_init(self, dataset, clip_model):
        vision, vision_mask, text, text_mask = dataset.full_item_features()

        vision = vision.unsqueeze(dim=1).to(self.args.device)
        vision_mask = vision_mask.unsqueeze(dim=1).to(self.args.device)
        text = text.unsqueeze(dim=1).to(self.args.device)
        text_mask = text_mask.unsqueeze(dim=1).to(self.args.device)

        with torch.no_grad():
            vision, vision_mask = clip_model(vision, vision_mask, mode="vision", format=self.args.train_vision_format)
            text, text_mask = clip_model(text, text_mask, mode="text", format=self.args.train_text_format)

        vision = vision.squeeze(dim=1)  # [N, D]
        vision_mask = vision_mask.squeeze(dim=1)  # [N]
        text = text.squeeze(dim=1)  # [N, D]
        text_mask = text_mask.squeeze(dim=1)  # [N]

        self.modality_embedding.weight_init(vision, vision_mask, text, text_mask)

    def forward(self,
                input_ids,
                vision=None,
                vision_mask=None,
                text=None,
                text_mask=None,
                mode=None):
        if mode == "mlm":
            return self.mlm(input_ids)
        elif mode == "pred":
            return self.prediction(input_ids)
        else:
            raise NotImplementedError("FusionEncoderForMaskedLM Forward Mode Error!")

    def modality_embedding_hidden(self):
        input_ids = torch.arange(3, self.modality_embedding.item_num + 3, dtype=torch.int, device=self.args.device)
        input_ids = input_ids.unsqueeze(dim=-1)

        vision, vision_mask, text, text_mask = self.modality_embedding(input_ids)
        embed, mask = self.input_transform(vision, vision_mask, text, text_mask)

        embed = self.embedding(input_ids, embed)
        inputs = torch.flatten(embed, 1, 2)
        hidden = self.encoder_blocks(inputs, ~mask.reshape(inputs.shape[0], -1))
        hidden = hidden.reshape(embed.shape)
        hidden = self.output_transform(hidden, mask)
        hidden = hidden[:, -1, :]

        return hidden

    def mlm(self, input_ids):
        ce_mask = torch.full_like(input_ids, self.ce_mask_ratio, dtype=torch.float) * (input_ids != 0)
        ce_mask = torch.bernoulli(ce_mask).type(torch.bool)
        ce_mask[:, -1] = 1
        masked_ids = input_ids.masked_fill(ce_mask, 1)
        replace_mask = torch.full_like(input_ids, 0.1, dtype=torch.float) * ce_mask
        replace_mask = torch.bernoulli(replace_mask).type(torch.bool)
        random_ids = torch.randint_like(input_ids, low=3, high=self.args.train_item_num + 2)
        masked_ids = torch.where(replace_mask, random_ids, masked_ids)
        restore_mask = torch.full_like(input_ids, 0.1, dtype=torch.float) * ce_mask
        restore_mask = torch.bernoulli(restore_mask).type(torch.bool)
        masked_ids = torch.where(restore_mask, input_ids, masked_ids)
        masked_ids[:, -1] = 1

        vision, vision_mask, text, text_mask = self.modality_embedding(masked_ids)
        embed, mask = self.input_transform(vision, vision_mask, text, text_mask)

        embed = self.embedding(masked_ids, embed)
        inputs = torch.flatten(embed, 1, 2)
        hidden = self.encoder_blocks(inputs, ~mask.reshape(inputs.shape[0], -1))
        hidden = hidden.reshape(embed.shape)
        hidden = self.output_transform(hidden, mask)
        hidden = torch.masked_select(hidden, ce_mask.unsqueeze(-1)).reshape(-1, hidden.shape[-1])

        embedding_hidden = self.modality_embedding_hidden()

        hidden = nn.functional.normalize(hidden, dim=-1)
        embedding_hidden = nn.functional.normalize(embedding_hidden, dim=-1)

        predict = torch.matmul(hidden, embedding_hidden.T) / self.args.contrastive_temperature  # [L, V]

        labels = torch.masked_select(input_ids - 3, ce_mask).reshape(-1)
        return predict, labels

    def prediction(self, input_ids):
        labels = input_ids[:, -1].detach()  # [N]
        input_ids = input_ids.clone()
        input_ids[:, -1] = 1

        vision, vision_mask, text, text_mask = self.modality_embedding(input_ids)
        embed, mask = self.input_transform(vision, vision_mask, text, text_mask)

        embed = self.embedding(input_ids, embed)
        inputs = torch.flatten(embed, 1, 2)
        hidden = self.encoder_blocks(inputs, ~mask.reshape(inputs.shape[0], -1))
        hidden = hidden.reshape(embed.shape)
        hidden = self.output_transform(hidden, mask)
        hidden = hidden[:, -1, :]

        embedding_hidden = self.modality_embedding_hidden()

        hidden = nn.functional.normalize(hidden, dim=-1)
        embedding_hidden = nn.functional.normalize(embedding_hidden, dim=-1)
        predict = torch.matmul(hidden, embedding_hidden.T)  # [N, V]

        return predict, labels - 3








