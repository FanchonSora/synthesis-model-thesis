import torch

MODALITIES = ["t1", "t1ce", "t2", "flair"]

def build_inputs_from_fixed_target(
    images: torch.Tensor,
    target_modality_id: torch.Tensor,
):
    B = images.shape[0]
    device = images.device
    modality_mask = torch.ones(B, 4, device=device, dtype=images.dtype)
    modality_mask.scatter_(1, target_modality_id.unsqueeze(1), 0.0)

    x_input = images.clone()
    batch_idx = torch.arange(B, device=device)
    x_input[batch_idx, target_modality_id] = 0.0

    x_dict = {
        "t1": x_input[:, 0:1],
        "t1ce": x_input[:, 1:2],
        "t2": x_input[:, 2:3],
        "flair": x_input[:, 3:4],
    }
    x_target_gt = images[batch_idx, target_modality_id].unsqueeze(1)  # [B,1,H,W,D]
    return x_dict, x_target_gt, modality_mask