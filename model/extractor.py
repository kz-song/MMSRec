import os
import clip
import torch
from torch import nn


class ClipEncoder(nn.Module):
    def __init__(self,
                 args,
                 **kwargs):
        super().__init__()
        self.args = args

        model, image_processor = clip.load(args.clip_model_path, device="cpu")
        self.model = model

    def save_pretrained(self, save_path):
        state_file = os.path.join(save_path, "clip_model.pth")
        state = {
            "clip": self.model.state_dict()
        }
        torch.save(state, state_file)

    def from_pretrained(self, load_path, map_location):
        state_file = os.path.join(load_path, "clip_model.pth")
        if not os.path.exists(state_file):
            return

        state = torch.load(state_file, map_location=map_location)
        self.model.load_state_dict(state["clip"])

    def forward(self, embeds, mask, mode, format):
        assert mode in ["vision", "text"]

        if mode == "vision":
            return self.vision_encode(embeds, mask, format)
        else:
            return self.text_encode(embeds, mask, format)

    def vision_encode(self, vision, vision_mask, format):
        if format == "mp4":
            N, L, F, C, H, W = vision.shape

            vision = self.model.encode_image(vision.reshape(-1, C, H, W))
            vision = vision.reshape(N, L, F, -1)  # [batch, seq, frames, dim]

        vision_features_sum = (vision * vision_mask.unsqueeze(-1)).sum(dim=-2)
        vision_mask_sum = vision_mask.sum(dim=-1, keepdim=True)
        vision_mask_sum = torch.maximum(vision_mask_sum, torch.ones_like(vision_mask_sum))
        vision_embeddings = vision_features_sum / vision_mask_sum  # [batch, seq, dim]

        vision_mask = (vision_mask.sum(dim=-1) > 0).to(vision_mask.device)
        # vision_embeddings     torch.tensor((batch, seq, dim))
        # vision_mask           torch.tensor((batch, seq))
        return vision_embeddings, vision_mask

    def text_encode(self, text, text_mask, format):
        if format != "txt":
            return text, text_mask

        N, L, T = text.shape

        text_features = self.model.encode_text(text.reshape(-1, T))
        text_embeddings = text_features.reshape(N, L, -1)

        # text_embeddings       torch.tensor((batch, seq, dim))
        # text_mask             torch.tensor((batch, seq))
        return text_embeddings, text_mask


