from pathlib import Path
import tqdm
import gc
from PIL import Image
import torch
import numpy as np
import schedulefree

from dataset import HypersimSegmentationDataset
from evaluate_sam2 import sample_result, get_mask_generator, generate_masks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# define losses
def loss_tree(A):
    # A is (M, P) with M masks and P pixels
    term1 = A @ A.T  # (M, M)
    term2 = (1 - A) @ A.T
    term3 = term2.T

    min_terms = torch.minimum(torch.minimum(term1, term2), term3)  # (M, M)
    min_terms.fill_diagonal_(0)
    return torch.mean(min_terms / A.shape[1])


def loss_tv(z, H, W):  # total variation loss
    z = z.view(z.shape[0], 1, H, W)
    return torch.mean(torch.abs(z[:, :, :, :-1] - z[:, :, :, 1:])) + torch.mean(
        +torch.abs(z[:, :, :-1, :] - z[:, :, 1:, :])
    )


def loss_reg(f, eps=5):
    return torch.mean(torch.relu(f**2 - eps**2))


def soft_max(values, dim, temperature):
    softmax_values = torch.nn.functional.softmax(values / temperature, dim=dim)
    # print(
    #     f"Softmax values mean: {softmax_values.mean().item()}, max: {softmax_values.max().item()}, min: {softmax_values.min().item()}"
    # )
    return torch.sum(values * softmax_values, dim=dim)


def loss_fit(z, gts, temperature=1.0, epsilon=1e-6):
    """
    Compute the differentiable approximation of the loss:
        (1 / N) * sum_i max_j IoU(g_i, z_j)

    Args:
        z (Tensor): Predicted probabilities of shape (M, L).
        gts (Tensor): Ground truth masks of shape (N, L).
        temperature (float): Temperature parameter for softmax.
        epsilon (float): Small constant for numerical stability.

    Returns:
        Tensor: Scalar loss value.
    """
    # Compute IoU matrix (N x M)
    intersection = torch.matmul(gts.float(), z.t())  # (N, M)
    gts_sum = gts.sum(dim=1, keepdim=True)  # (N, 1)
    z_sum = z.sum(dim=1, keepdim=True)  # (M, 1)
    union = gts_sum + z_sum.t() - intersection  # (N, M)
    iou = intersection / (union + epsilon)  # (N, M)

    # Compute the approximate maximum IoU for each ground truth
    # approx_max_iou = soft_max(iou, dim=1, temperature=temperature)
    approx_max_iou = torch.max(iou, dim=1).values

    # Compute the average over all ground truths
    loss = 1 - approx_max_iou.mean()
    # print(f"Max Iou: {approx_max_iou.mean().item()}, max: {approx_max_iou.max().item()}, min: {approx_max_iou.min().item()}")
    return loss


def loss_area(z):  # big areas are good, small areas are bad
    return torch.log(torch.sum(z, dim=1)).mean()


def main(
    max_sam_masks=1000,
    M=32,
    lr=1,
    steps=10,
    temp=1,
    w_fit=0,
    w_area=0,
    w_reg=0,
    w_tv=0,
    w_tree=0,
    outdir="vis",
):
    log_dict = locals() | {"file": __file__}
    # get script name and parameters
    outdir = Path(outdir)
    (outdir / "live").mkdir(exist_ok=True, parents=True)
    torch.random.manual_seed(0)
    dataset = HypersimSegmentationDataset()
    image, labels = dataset[0]
    # visualize labels
    Image.fromarray((labels / labels.max() * (255**2 - 1)).astype(np.uint16)).save(
        outdir / "labels.png"
    )

    # build mask generator
    mask_generator = get_mask_generator(
        points_per_side=32,
        stability_score_thresh=0,
        box_nms_thresh=0.95,
        pred_iou_thresh=0,
        points_per_batch=32,
        crop_nms_thresh=1,
        crop_n_layers=0,
        device=device,
    )

    # in this script, gts or new_labels are to evaluate, while `masks` guide the optimization

    masks, logits, predicted_ious, stability_scores = generate_masks(
        None, image, mask_generator
    )
    masks = torch.from_numpy(masks).to(device)
    # free memory
    del mask_generator
    del logits
    torch.cuda.empty_cache()
    gc.collect()
    print("-" * 80)
    print("-" * 80)
    print("-" * 80)
    mean_scores = torch.from_numpy((stability_scores + predicted_ious) / 2)
    selected_indices = torch.argsort(mean_scores, descending=True)[:max_sam_masks]
    masks = masks[selected_indices]
    mean_scores = mean_scores[selected_indices]

    # build ground truth masks
    gts = []
    for label in np.unique(labels):
        mask = labels == label
        gts.append(mask)
    gts = torch.from_numpy(np.array(gts)).to(device)
    N, H, W = gts.shape
    gts = gts.view(N, H * W)
    gts = gts[gts.sum(dim=1) > 63]  # should be at least an 8x8 patch
    masks = masks.view(-1, H * W)

    new_labels = np.zeros_like(labels)
    print("Visualizing ground truth masks...")
    for i, mask in tqdm.tqdm(enumerate(gts)):
        new_labels[mask.view(H, W).cpu().numpy()] = i
    Image.fromarray(
        (new_labels / new_labels.max() * (255**2 - 1)).astype(np.uint16)
    ).save(outdir / "gts.png")

    print("Optimizing...")
    # build random logits
    X = (
        (torch.randn((M, H * W))).to(device).requires_grad_(True)
    )  # tensor to be optimized of shape (n_masks, H, W)
    # optimizer = schedulefree.AdamWScheduleFree([X], lr=lr)
    optimizer = schedulefree.RAdamScheduleFree([X], lr=lr)
    optimizer.train()
    loss_history = []

    # optimize
    for i in range(steps):
        optimizer.zero_grad()
        # compute z
        z = torch.sigmoid(X)
        # compute losses
        lfit = loss_fit(z, masks, temperature=temp) if w_fit > 0 else 0
        larea = loss_area(z) if w_area > 0 else 0
        lreg = loss_reg(X) if w_reg > 0 else 0
        ltv = loss_tv(X, H, W)
        ltree = loss_tree(z) if w_tree > 0 else 0
        loss = (
            lfit * w_fit + larea * w_area + lreg * w_reg + ltv * w_tv + ltree * w_tree
        )
        # optimize
        loss.backward()
        # print(
        #     f"Grad mean: {X.grad.mean().item()}, max: {X.grad.max().item()}, min: {X.grad.min().item()}"
        # )
        optimizer.step()
        loss_history.append(loss.item())

        if i % 10 == 0 or i < 20:
            print(f"Step {i}, loss={loss.item()}", end="\r")
            z = torch.sigmoid(X)
            classes = z.argmax(dim=0).view(H, W).cpu().numpy()
            Image.fromarray(
                (classes / classes.max() * (256**1 - 1)).astype(np.uint8)
            ).save(outdir / "live" / f"classes_live_{i}.png")
    print(f"Step {i}, loss={loss.item()}")

    # score and visualize
    z = torch.sigmoid(X)
    classes = z.argmax(dim=0).view(H, W).cpu().numpy()
    pred_masks = (
        (z.view(M, H, W) == z.max(dim=0).values.view(1, H, W)).detach().cpu().numpy()
    )
    result = sample_result(pred_masks, new_labels, device)

    selected_indices = torch.argsort(mean_scores, descending=True)[:M]
    sam_res = sample_result(
        masks[selected_indices].view(-1, H, W).cpu().numpy(), new_labels, device
    )
    print("adjusted mIoU:", result["mIoU"])
    print("sam mIoU:", sam_res["mIoU"])
    log_dict["sam_mIoU"] = sam_res["mIoU"]
    log_dict["adjusted_mIoU"] = result["mIoU"]
    import json

    with open(outdir / "log.json", "w") as f:
        json.dump(log_dict, f)

    Image.fromarray((classes / classes.max() * (256**1 - 1)).astype(np.uint8)).save(
        outdir / "classes.png"
    )
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(loss_history)
    plt.savefig(outdir / "loss.png")
    plt.close()


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
