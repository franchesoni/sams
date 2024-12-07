import numpy as np
import sys
from PIL import Image
import h5py
from pathlib import Path

# miport pytorch ds
from torch.utils.data import Dataset


class HypersimSegmentationDataset(Dataset):
    """One frame for one camera trajectory for each scene paired with the ground truth object ids."""

    def __init__(self, root_dir="downloads", transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        print("Finding images and masks...")
        images = sorted(
            self.root_dir.glob(
                "ai_*/images/scene_cam_00_final_preview/frame.0000.color.jpg"
            )
        )
        masks = sorted(
            self.root_dir.glob(
                "ai_*/images/scene_cam_00_geometry_hdf5/frame.0000.render_entity_id.hdf5"
            )
        )
        self.images, self.masks = [], []
        for mask_path in masks:
            img_path = Path(
                str(mask_path).split("_geometry")[0]
                + "_final_preview/frame.0000.color.jpg"
            )
            if img_path in images:
                self.images.append(img_path)
                self.masks.append(mask_path)
        print("Found {} images and masks.".format(len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        labels = h5py.File(self.masks[idx], "r")["dataset"][:]
        if self.transform:
            image, labels = self.transform(image, labels)
        return image, labels

    def visualize(self, idx):
        print("Visualizing sample {}...".format(idx))
        image, labels = self.__getitem__(idx)
        self.show_sample(image, labels)
        return image, labels

    def show_sample(self, image, labels):
        Path("vis").mkdir(exist_ok=True)
        image.save("vis/image.png")
        Image.fromarray(
            (labels / labels.max() * (256 * 256 - 1)).astype(np.uint16)
        ).save("vis/labels.png")


if __name__ == "__main__":
    ind = int(sys.argv[1])
    ds = HypersimSegmentationDataset(root_dir="downloads")
    ds.visualize(ind)
