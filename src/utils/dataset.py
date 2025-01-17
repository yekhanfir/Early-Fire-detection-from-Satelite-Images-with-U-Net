import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from src.utils.image_utils import normalize_image

class SegmentationDataset(Dataset):
    def __init__(self, dataset_dict, folds, channels=(9, 10, 11), sample_ratio=0.1):
        self.images = []
        self.masks = []
        self.sample_ratio = sample_ratio

        for fold in folds:
            print(f"Processing fold {fold}")
            for idx in tqdm(range(len(dataset_dict[fold]))):
                self.images.append(
                    np.stack(
                        [
                            normalize_image(
                                np.array(dataset_dict[fold][idx]['post_fire'])[:, :, c]
                            ) for c in channels
                        ],
                        axis=-1
                    )
                )
                self.masks.append(np.array(dataset_dict[fold][idx]['mask']))

    def randomly_sample(self, mask):
        """
        Randomly sample 10% of the positive and 10% of the negative area (1s and 0s in the mask)
        and set unselected points to -1.

        Args:
            image (np.ndarray): Input image of shape (512, 512, 3).
            mask (np.ndarray): Binary mask of shape (512, 512, 1).
            sample_ratio (float): Proportion of positive and negative areas to sample.

        Returns:
            np.ndarray: Updated mask with unselected points set to -1.
            tuple: Indices of sampled points in the (H, W) format for both positive and negative areas.
        """
        # Ensure the mask is a single channel
        if len(mask.shape) == 3:
            mask = mask.squeeze(-1)

        # Find the indices of positive and negative areas (where mask value is 1 and 0)
        positive_indices = np.where(mask == 1)
        negative_indices = np.where(mask == 0)

        num_positive_pixels = len(positive_indices[0])
        num_negative_pixels = len(negative_indices[0])

        # Determine the number of positive and negative pixels to sample
        positive_sample_size = int(self.sample_ratio * num_positive_pixels)
        negative_sample_size = int(self.sample_ratio * num_negative_pixels)

        # Randomly select indices from the positive and negative areas
        sampled_positive_indices = np.random.choice(
            num_positive_pixels,
            size=positive_sample_size,
            replace=False
        )
        sampled_negative_indices = np.random.choice(
            num_negative_pixels,
            size=negative_sample_size,
            replace=False
        )

        # Create a mask to update unselected points to -1
        updated_mask = np.full_like(mask, -1, dtype=np.int32)

        # Set the sampled positive and negative indices back to 1 and 0 in the updated mask
        selected_positives = (
            positive_indices[0][sampled_positive_indices],
            positive_indices[1][sampled_positive_indices]
        )
        selected_negatives = (
            negative_indices[0][sampled_negative_indices],
            negative_indices[1][sampled_negative_indices]
        )

        updated_mask[selected_positives] = 1
        updated_mask[selected_negatives] = 0

        # Concatenate the selected positive and negative indices into one array
        combined_selected_indices = np.concatenate([selected_positives, selected_negatives], axis=1)

        return updated_mask, combined_selected_indices

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        updated_mask, _ = self.randomly_sample(mask, self.sample_ratio)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(updated_mask, dtype=torch.long)
