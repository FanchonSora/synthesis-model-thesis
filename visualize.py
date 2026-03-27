import os
import torch
import matplotlib.pyplot as plt
from synthesis_utils import build_inputs_from_fixed_target, MODALITIES

def _to_cpu_float(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu().float()

def _norm_2d_for_display(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x

def _extract_slice(
    volume_3d: torch.Tensor,
    plane: str = "axial",
    slice_idx: int = None,
) -> torch.Tensor:
    H, W, D = volume_3d.shape
    if plane == "axial":
        idx = D // 2 if slice_idx is None else max(0, min(D - 1, slice_idx))
        img = volume_3d[:, :, idx]
    elif plane == "coronal":
        idx = W // 2 if slice_idx is None else max(0, min(W - 1, slice_idx))
        img = volume_3d[:, idx, :]
    elif plane == "sagittal":
        idx = H // 2 if slice_idx is None else max(0, min(H - 1, slice_idx))
        img = volume_3d[idx, :, :]
    else:
        raise ValueError(f"Unsupported plane: {plane}. Choose from axial/coronal/sagittal.")
    return img

def _save_single_synthesis_figure(
    case_id: str,
    target_modality: str,
    plane: str,
    gt_img: torch.Tensor,
    gen_img: torch.Tensor,
    available_imgs: dict,
    save_path: str,
):
    error_img = torch.abs(gen_img - gt_img)
    gt_show = _norm_2d_for_display(gt_img)
    gen_show = _norm_2d_for_display(gen_img)
    err_show = _norm_2d_for_display(error_img)
    available_names = list(available_imgs.keys())
    n_inputs = len(available_names)
    ncols = n_inputs + 3  # inputs + GT + Gen + Error
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]
    col = 0
    for mod_name in available_names:
        img = _norm_2d_for_display(available_imgs[mod_name])
        axes[col].imshow(img, cmap="gray")
        axes[col].set_title(f"Input: {mod_name.upper()}")
        axes[col].axis("off")
        col += 1
    axes[col].imshow(gt_show, cmap="gray")
    axes[col].set_title(f"GT: {target_modality.upper()}")
    axes[col].axis("off")
    col += 1
    axes[col].imshow(gen_show, cmap="gray")
    axes[col].set_title(f"Synth: {target_modality.upper()}")
    axes[col].axis("off")
    col += 1
    axes[col].imshow(err_show, cmap="hot")
    axes[col].set_title("|Synth - GT|")
    axes[col].axis("off")
    plt.suptitle(
        f"Case: {case_id} | Missing: {target_modality.upper()} | Plane: {plane}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

@torch.no_grad()
def save_raw_slice_panels(
    test_loader,
    save_dir: str,
    num_cases: int = 5,
    plane: str = "axial",
    slice_idx: int = None,
):
    os.makedirs(save_dir, exist_ok=True)
    saved_cases = 0
    for batch in test_loader:
        images = batch["image"]
        case_ids = batch.get("case_id", None)
        if torch.is_tensor(images):
            images = images.cpu()
        B = images.shape[0]
        for b in range(B):
            if saved_cases >= num_cases:
                return
            case_id = case_ids[b] if case_ids is not None else f"case_{saved_cases:04d}"
            vols = images[b]  # [4,H,W,D]
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            for i, mod_name in enumerate(MODALITIES):
                vol_3d = vols[i].float()
                img_2d = _extract_slice(vol_3d, plane=plane, slice_idx=slice_idx)
                img_2d = _norm_2d_for_display(img_2d)

                axes[i].imshow(img_2d, cmap="gray")
                axes[i].set_title(mod_name.upper())
                axes[i].axis("off")
            plt.suptitle(f"Case: {case_id} | Plane: {plane}", fontsize=12)
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, f"{str(case_id)}_raw_{plane}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
            saved_cases += 1

@torch.no_grad()
def visualize_synthesis_results(
    model,
    test_loader,
    device,
    save_dir: str,
    num_cases: int = 5,
    num_infer_steps: int = 50,
    plane: str = "axial",
    slice_idx: int = None,
):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    saved_cases = 0
    for batch in test_loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)
        images = batch["image"]        # [B,4,H,W,D]
        domain_id = batch["domain_id"] # [B]
        case_ids = batch.get("case_id", None)
        B = images.shape[0]
        for b in range(B):
            if saved_cases >= num_cases:
                return
            case_id = case_ids[b] if case_ids is not None else f"case_{saved_cases:04d}"
            image_b = images[b:b+1]      # [1,4,H,W,D]
            domain_b = domain_id[b:b+1]  # [1]
            case_dir = os.path.join(save_dir, str(case_id))
            os.makedirs(case_dir, exist_ok=True)
            for mod_id, mod_name in enumerate(MODALITIES):
                target_modality_id = torch.tensor([mod_id], device=device, dtype=torch.long)
                x_dict, x_target_gt, modality_mask = build_inputs_from_fixed_target(
                    image_b, target_modality_id
                )
                outputs = model(
                    x_dict=x_dict,
                    x_target_gt=None,
                    target_modality_name=None,
                    target_modality_id=target_modality_id,
                    modality_mask=modality_mask,
                    domain_id=domain_b,
                    t=None,
                    unconditional=False,
                    num_infer_steps=num_infer_steps,
                )
                x_hat = outputs["x_hat"]  # [1,1,H,W,D]
                gt_3d = _to_cpu_float(x_target_gt[0, 0])  # [H,W,D]
                gen_3d = _to_cpu_float(x_hat[0, 0])       # [H,W,D]
                gt_2d = _extract_slice(gt_3d, plane=plane, slice_idx=slice_idx)
                gen_2d = _extract_slice(gen_3d, plane=plane, slice_idx=slice_idx)
                available_imgs = {}
                for in_mod_name in MODALITIES:
                    if in_mod_name == mod_name:
                        continue
                    vol_3d = _to_cpu_float(x_dict[in_mod_name][0, 0])
                    available_imgs[in_mod_name] = _extract_slice(
                        vol_3d, plane=plane, slice_idx=slice_idx
                    )
                save_path = os.path.join(
                    case_dir,
                    f"{str(case_id)}_missing_{mod_name}_{plane}.png"
                )
                _save_single_synthesis_figure(
                    case_id=str(case_id),
                    target_modality=mod_name,
                    plane=plane,
                    gt_img=gt_2d,
                    gen_img=gen_2d,
                    available_imgs=available_imgs,
                    save_path=save_path,
                )
            saved_cases += 1