import numpy as np
import pandas as pd
import os
import cv2
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class MultiLabelDataset(Dataset):

    def __init__(self, df, image_size: tuple, transform=None) -> None:

        """
        Dataset for multi-label classification.
        
        Args:
        - df (DataFrame): DataFrame contain image and label information.
        - image_root (str): image path.
        - transform (callable, optional): Transform image.
        """

        self.df = df
        self.image_size = image_size
        self.transform = transform
        self.labels = self.df.iloc[:, 3:].fillna(0.0).values.astype(np.float32)  # Replace NaN with 0.0
        self.path = self.df["path"].tolist()  # Column containing the path to the image

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx: int) -> dict:

        img_path = os.path.join(self.path[idx])
        
        # Read image, handle error if image cannot be opened
        try:
            image = Image.open(img_path).convert("RGB")  # Convert to RGB for consistency
        except Exception as e:
            print(f"Error loading image {names}: {e}")
            return None
        
        image = np.array(image)
        image = cv2.resize(image, self.image_size)  # Resize to model's standard size
        image = image / 255.0  # Normalize to [0, 1]
        image = np.transpose(image, (2, 0, 1))  # Move color channel to top (C, H, W)

        label = self.labels[idx]

        sample = {"image": image, "label": label}
        return sample


def load_data(csv_path: str) -> tuple:

    """
    Read CSV file, process data, and split into train, val, test sets.
    
    Args:
    - csv_path (str): CSV file path.
    - image_root (str): Folder path containing images.
    
    Returns:
    - train_df, val_df, test_df (DataFrame): The data sets have been split.
    """

    df = pd.read_csv(csv_path)

    # Rename columns if necessary
    df.columns = ["dataset", "filename", "path"] + df.columns[3:].tolist()

    # Filter data by set
    train_df = df[df["dataset"] == "train"].reset_index(drop=True)
    val_df = df[df["dataset"] == "val"].reset_index(drop=True)
    test_df = df[df["dataset"] == "test"].reset_index(drop=True)

    return train_df, val_df, test_df


if __name__ == "__main__":

    csv_path = "Variant-b(MultiLabel Classification)/Multi-Label dataset.csv"
    image_size = (224, 224)

    train_df, val_df, test_df = load_data(csv_path)

    # Create dataset
    train_dataset = MultiLabelDataset(train_df, image_size)
    val_dataset = MultiLabelDataset(val_df, image_size)
    test_dataset = MultiLabelDataset(test_df, image_size)

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Data checking
    sample = next(iter(train_loader))
    print("Image shape:", sample["image"].shape)
    print("Label shape:", sample["label"].shape)

    # for i, train in enumerate(train_loader):

    #     img = train["image"].squeeze().permute(1, 2, 0).numpy()
    #     l = train["label"].tolist()[0]

    #     if i > 1:
    #         break

    #     print(l)
    #     plt.title(l)
    #     plt.imshow(img)
    #     plt.show()
