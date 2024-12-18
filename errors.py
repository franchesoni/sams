# %% [markdown]
# # Analyze SAM2 failure modes over entire dataset

# %% [markdown]
# ## Preliminaries

# %%
import tqdm
import os
import json
from PIL import Image
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from scipy.stats import spearmanr

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator, rle_to_mask

from dataset import HypersimSegmentationDataset

def minmax(x):
    amin, amax = x.min(), x.max()
    if amin == amax:
        return x
    return (x - amin) / (amax - amin)

def get_mask_generator(
    points_per_side,
    stability_score_thresh,
    box_nms_thresh,
    pred_iou_thresh,
    points_per_batch,
    crop_nms_thresh,
    crop_n_layers,
    device,
):
    sam2_checkpoint = (
        "/home/franchesoni/sam2/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt"
    )
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_model = build_sam2(
        model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False
    )
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=points_per_side,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
        pred_iou_thresh=pred_iou_thresh,
        points_per_batch=points_per_batch,
        crop_nms_thresh=crop_nms_thresh,
        crop_n_layers=crop_n_layers,
    )
    return mask_generator

def sample_result(masks, labels, device):
    """
    masks: (M, H, W), np.ndarray
    labels: (H, W), np.ndarray
    """
    H, W = labels.shape
    masks = torch.from_numpy(masks).reshape(-1, H * W).to(device)  # (M, HW)
    labels = torch.from_numpy(labels).reshape(1, H * W).to(device)
    max_ious = []
    unique_labels = torch.unique(labels)
    for label in unique_labels:
        if label == 0:
            continue
        label_mask = labels == label  # (1, HW)
        if label_mask.any():  # If not empty
            intersection = (masks & label_mask).sum(dim=1)  # (M,)
            union = (masks | label_mask).sum(dim=1)  # (M,)
            ious = intersection / union  # (M,)
            max_iou = ious.max()
            max_ious.append(max_iou.item())
    return {
        "L": len(unique_labels) - (1 if 0 in unique_labels else 0),
        "M": len(masks),
        "max_ious": max_ious,
        "mIoU": np.mean(max_ious) if len(max_ious) > 0 else 0.0,
    }

def build_new_labels(labels, min_size=64, device='cpu'):
    # build ground truth masks
    gts = []
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = labels == label
        gts.append(mask)
    if len(gts) == 0:
        return labels
    gts = np.array(gts)
    gts = torch.from_numpy(gts).to(device)
    N, H, W = gts.shape
    gts = gts.view(N, H * W)
    # filter out small masks
    gts = gts[gts.sum(dim=1) >= min_size]
    new_labels = np.zeros((H, W), dtype=labels.dtype)
    for i, mask in enumerate(gts):
        new_labels[mask.view(H, W).cpu().numpy()] = i+1
    return new_labels

def generate_masks(path_to_image, image, mask_generator):
    assert (path_to_image is None and isinstance(image, Image.Image)) or (
        isinstance(path_to_image, str) and image is None
    )
    if image is None:
        image = np.array(Image.open(path_to_image).convert("RGB"))
    else:
        image = np.array(image)
    mask_data = mask_generator._generate_masks(image)
    stability_scores = mask_data["stability_score"]
    logits = mask_data["low_res_masks"]
    ious = mask_data["iou_preds"]
    mask_data["segmentations"] = [
        rle_to_mask(rle) for rle in mask_data["rles"]
    ]  # masks
    masks = np.array(mask_data["segmentations"])
    return masks, logits, ious, stability_scores

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # use bfloat16 for the entire run
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Set parameters
    points_per_side=32
    stability_score_thresh=0
    box_nms_thresh=0.90
    pred_iou_thresh=0.0
    points_per_batch=32
    crop_nms_thresh=1
    crop_n_layers=0

    mask_generator = get_mask_generator(
        points_per_side=points_per_side,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
        pred_iou_thresh=pred_iou_thresh,
        points_per_batch=points_per_batch,
        crop_nms_thresh=crop_nms_thresh,
        crop_n_layers=crop_n_layers,
        device=device,
    )

    eval_ds = HypersimSegmentationDataset()
    out_dir = Path("evaluation_results")
    out_dir.mkdir(exist_ok=True, parents=True)

    # The values of max_n_masks we want to evaluate
    max_n_maskss = [2, 4, 16, 64, 256, 512, 1024]
    min_target_size = 256

    # We'll store global results as well
    global_results = []

    for sample_index in tqdm.trange(len(eval_ds), desc="Evaluating dataset"):
        image, labels = eval_ds[sample_index]
        H, W = image.height, image.width
        masks, logits, predicted_ious, stability_scores = generate_masks(
            None, image, mask_generator
        )

        mean_scores = (stability_scores + predicted_ious) / 2
        mean_scores = torch.from_numpy(mean_scores)
        blabels = build_new_labels(labels, min_size=min_target_size, device=device)

        # Evaluate performance with varying number of masks
        n_masks_list = []
        mious_list = []
        for max_n_masks in max_n_maskss:
            selected_indices = torch.argsort(mean_scores, descending=True)[:max_n_masks]
            smasks = masks[selected_indices.cpu().numpy()]
            res = sample_result(smasks, blabels, device)
            mious_list.append(res['mIoU'])
            n_masks_list.append(len(smasks))

        # Compute region properties correlation
        rprops = regionprops(blabels)
        attributes = ['area', 'perimeter', 'equivalent_diameter', 'solidity', 'eccentricity', 'extent', 'convex_area']
        maxious = res['max_ious']
        correlations = {}
        if len(rprops) == len(maxious) and len(maxious) > 0:
            for attr in attributes:
                attr_values = [rprops[i][attr] for i in range(len(rprops))]
                scorr, _ = spearmanr(maxious, attr_values)
                correlations[attr] = scorr

        # Identify hardest masks
        # We'll define "hard" as those with lowest IoU
        if len(maxious) > 0:
            hard_target_indices = np.argsort(maxious)[:10] + 1
        else:
            hard_target_indices = []

        # Create output directory for this sample
        sample_dir = out_dir / f"sample_{sample_index:05d}"
        sample_dir.mkdir(exist_ok=True, parents=True)

        # Save hard target label images
        for label_id in hard_target_indices:
            plt.figure(figsize=(5, 5))
            plt.title(f'Hard target label {label_id}')
            plt.imshow(blabels == label_id)
            plt.axis('off')
            plt.savefig(sample_dir / f"hard_label_{label_id}.png", dpi=150, bbox_inches='tight')
            plt.close()

        # Save histogram of errors
        if len(maxious) > 0:
            plt.figure()
            plt.title('Max IoU Histogram')
            plt.hist(maxious, bins=20)
            plt.xlabel("Max IoU")
            plt.ylabel("Count")
            plt.savefig(sample_dir / "max_iou_histogram.png", dpi=150, bbox_inches='tight')
            plt.close()

        # Save performance curve
        plt.figure(figsize=(10, 6))
        plt.plot(n_masks_list, mious_list, marker='o')
        plt.xlabel('Number of masks')
        plt.ylabel('mIoU')
        plt.title('mIoU vs Number of Masks')
        plt.grid(True)
        plt.savefig(sample_dir / "mIoU_vs_n_masks.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Save numeric results and correlations to a json or txt file
        results_dict = {
            "sample_index": int(sample_index),
            "mIoU_vs_n_masks": {str(int(n)): float(m) for n, m in zip(n_masks_list, mious_list)},
            "correlations": {k: float(v) for k, v in correlations.items()},
            "mean_mIoU": float(np.mean(mious_list) if len(mious_list) > 0 else 0.0),
            "num_objects": int(blabels.max()),
            "hard_labels": [int(x) for x in hard_target_indices],
        }

        with open(sample_dir / "results.json", "w") as f:
            json.dump(results_dict, f, indent=2)

        global_results.append(results_dict)

    # Optionally, save global results
    with open(out_dir / "global_results.json", "w") as f:
        json.dump(global_results, f, indent=2)
