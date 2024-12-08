import tqdm
import os
from PIL import Image
from pathlib import Path
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator, rle_to_mask

from dataset import HypersimSegmentationDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


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


def main(
    points_per_side=32,
    stability_score_thresh=0.75,
    box_nms_thresh=1,
    pred_iou_thresh=0.75,
    points_per_batch=32,
    crop_nms_thresh=1,
    crop_n_layers=0,
):

    mask_generator = get_mask_generator(
        points_per_side=points_per_side,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
        pred_iou_thresh=pred_iou_thresh,
        points_per_batch=points_per_batch,
        crop_nms_thresh=crop_nms_thresh,
        crop_n_layers=crop_n_layers,
    )

    eval_ds = HypersimSegmentationDataset()
    res_per_sample = []
    for ind, (image, labels) in tqdm.tqdm(enumerate(eval_ds), total=len(eval_ds)):
        H, W = image.height, image.width
        masks, logits, predicted_ious, stability_scores = generate_masks(
            None, image, mask_generator
        )
        masks = torch.from_numpy(masks).reshape(-1, H * W).to(device)  # (M, HW)
        labels = torch.from_numpy(labels).reshape(1, H * W).to(device)
        max_ious = []
        for label in torch.unique(labels):
            label_mask = labels == label  # (1, HW)
            if label_mask.any():  # If not empty
                intersection = (masks & label_mask).sum(dim=1)  # (M,)
                union = (masks | label_mask).sum(dim=1)  # (M,)
                ious = intersection / union  # (M,)
                max_iou = ious.max()
                max_ious.append(max_iou.item())
                # print(
                #     f"Max IoU for label {label.item()}: {max_iou.item()}, average: {ious.mean().item()}",
                #     end="\r",
                # )
        res_per_sample.append(
            {
                "L": len(torch.unique(labels)),
                "M": len(masks),
                "max_ious": max_ious,
                "mIoU": ious.mean().item(),
            }
        )

    np.save(
        "res_per_sample.npy",
        {
            "config": {
                "points_per_side": points_per_side,
                "stability_score_thresh": stability_score_thresh,
                "box_nms_thresh": box_nms_thresh,
                "pred_iou_thresh": pred_iou_thresh,
                "points_per_batch": points_per_batch,
                "crop_nms_thresh": crop_nms_thresh,
                "crop_n_layers": crop_n_layers,
            },
            "results": res_per_sample,
        },
    )
    print("end")


def generate_masks(path_to_image, image, mask_generator):
    assert (path_to_image is None and type(image) == Image.Image) or (
        type(path_to_image) == str and image is None
    )
    Path("logits").mkdir(exist_ok=True)
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
    from fire import Fire

    Fire(main)