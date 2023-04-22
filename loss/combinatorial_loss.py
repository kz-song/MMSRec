import torch
from loss.norm_softmax_loss import NormSoftmaxLoss


class CombinatorialLoss(object):
    def __init__(self,
                 args,
                 model,
                 **kwargs):
        self.args = args
        self.model = model
        self.loss = NormSoftmaxLoss(args, **args)

        self.fusion_embed_dim = args.fusion_embed_dim

    def __call__(self, input_ids, vision, vision_mask, text, text_mask):
        v_out1, v_mask1 = self.model(input_ids, vision, vision_mask, text, torch.zeros_like(text_mask))
        v_out1 = v_out1.reshape(-1, self.fusion_embed_dim)
        t_out1, t_mask1 = self.model(input_ids, vision, torch.zeros_like(vision_mask), text, text_mask)
        t_out1 = t_out1.reshape(-1, self.fusion_embed_dim)

        v_out2, v_mask2 = self.model(input_ids, vision, vision_mask, text, torch.zeros_like(text_mask))
        v_out2 = v_out2.reshape(-1, self.fusion_embed_dim)
        t_out2, t_mask2 = self.model(input_ids, vision, torch.zeros_like(vision_mask), text, text_mask)
        t_out2 = t_out2.reshape(-1, self.fusion_embed_dim)

        vt_out1, vt_mask1 = self.model(input_ids, vision, vision_mask, text, text_mask)
        vt_out1 = vt_out1.reshape(-1, self.fusion_embed_dim)
        vt_out2, vt_mask2 = self.model(input_ids, vision, vision_mask, text, text_mask)
        vt_out2 = vt_out2.reshape(-1, self.fusion_embed_dim)

        loss_sum = torch.zeros(1).to(self.args.device)
        for out1, mask1, out2, mask2 in [
            (v_out1, v_mask1, v_out2, v_mask2),
            (v_out1, v_mask1, t_out1, t_mask1),
            (t_out1, t_mask1, t_out2, t_mask2),
            (vt_out1, vt_mask1, vt_out2, vt_mask2),
        ]:
            loss_sum += self.loss(out1, mask1, out2, mask2)
        loss_sum /= 4

        return loss_sum










