import tqdm
from PIL import Image
import torch
import numpy as np
import schedulefree

from dataset import HypersimSegmentationDataset
from evaluate_sam2 import sample_result

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# define losses
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


def main(M=32, lr=1, steps=10, temp=1, w_fit=0, w_area=0, w_reg=0, w_tv=0):
    torch.random.manual_seed(0)
    dataset = HypersimSegmentationDataset()
    image, labels = dataset[0]
    # visualize labels
    Image.fromarray((labels / labels.max() * (255**2 - 1)).astype(np.uint16)).save(
        "vis/labels.png"
    )

    # build ground truth masks
    gts = []
    for label in np.unique(labels):
        mask = labels == label
        gts.append(mask)
    gts = torch.from_numpy(np.array(gts)).to(device)
    N, H, W = gts.shape
    gts = gts.view(N, H * W)
    gts = gts[gts.sum(dim=1) > 63]  # should be at least an 8x8 patch
    new_labels = np.zeros_like(labels)
    print("Visualizing ground truth masks...")
    for i, mask in tqdm.tqdm(enumerate(gts)):
        new_labels[mask.view(H, W).cpu().numpy()] = i
    Image.fromarray(
        (new_labels / new_labels.max() * (255**2 - 1)).astype(np.uint16)
    ).save("vis/gts.png")

    print("Optimizing...")
    # build random logits
    X = (
        (torch.randn((M, H * W))).to(device).requires_grad_(True)
    )  # tensor to be optimized of shape (n_masks, H, W)
    # optimizer = schedulefree.AdamWScheduleFree([X], lr=lr)
    optimizer = schedulefree.RAdamScheduleFree([X], lr=lr)
    optimizer.train()

    # optimize
    for i in range(steps):
        optimizer.zero_grad()
        # compute z
        z = torch.softmax(X / temp, dim=0)
        # compute losses
        lfit = loss_fit(z, gts, temperature=temp) if w_fit > 0 else 0
        larea = loss_area(z) if w_area > 0 else 0
        lreg = loss_reg(X) if w_reg > 0 else 0
        ltv = loss_tv(X, H, W)
        loss = lfit * w_fit + larea * w_area + lreg * w_reg + ltv * w_tv
        # optimize
        loss.backward()
        # print(
        #     f"Grad mean: {X.grad.mean().item()}, max: {X.grad.max().item()}, min: {X.grad.min().item()}"
        # )
        optimizer.step()

        if i % 10 == 0 or i < 20:
            print(f"Step {i}, loss={loss.item()}", end="\r")
            z = torch.softmax(X / temp, dim=0)
            classes = z.argmax(dim=0).view(H, W).cpu().numpy()
            Image.fromarray(
                (classes / classes.max() * (256**1 - 1)).astype(np.uint8)
            ).save(f"vis/classes_live_{i}.png")
    print(f"Step {i}, loss={loss.item()}")

    # score and visualize
    z = torch.softmax(X / temp, dim=0)
    classes = z.argmax(dim=0).view(H, W).cpu().numpy()
    masks = (
        (z.view(M, H, W) == z.max(dim=0).values.view(1, H, W)).detach().cpu().numpy()
    )
    result = sample_result(masks, new_labels, device)
    print("mIoU:", result["mIoU"])
    Image.fromarray((classes / classes.max() * (256**1 - 1)).astype(np.uint8)).save(
        "vis/classes.png"
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)

# python oracle_optim.py --M=256 --w_fit=1 --temp=1 --lr=10 --steps=1000 --w_reg=0.1
