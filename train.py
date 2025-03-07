import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, VGG16_BN_Weights, Inception_V3_Weights, GoogLeNet_Weights
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import StepLR
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from data_processing import MultiLabelDataset, load_data


class Train:
    
    def __init__(self, models_list: list, csv_path: str, num_epochs: int, criterion, optimizer: str,
                 scheduler: int, device, image_size: tuple, batch_size: int):

        self.models_list = models_list
        self.csv_path = csv_path
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.device = device
        self.image_size = image_size
        self.batch_size = batch_size

        image_root = "Variant-b(MultiLabel Classification)/"
        train_df, val_df, test_df = load_data(self.csv_path, image_root)

        self.train_dataset = MultiLabelDataset(train_df, image_root, self.image_size)
        self.val_dataset = MultiLabelDataset(val_df, image_root, self.image_size)
        self.test_dataset = MultiLabelDataset(test_df, image_root, self.image_size)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        if optimizer == "SGD":
            self.opt_func = lambda model: SGD(model.parameters(), lr=0.001, momentum=0.9)
        elif optimizer == "AdamW":
            self.opt_func = lambda model: AdamW(model.parameters(), lr=0.001)
        else:
            raise ValueError("Invalid optimizer, only 'SGD' or 'AdamW' is supported.")

        self.scheduler_step = scheduler

    def train(self):
        for base_model in self.models_list:
            print(f"\n🔹 Training model: {type(base_model).__name__}")

            model = base_model.to(self.device)
            optimizer = self.opt_func(model)
            scheduler = StepLR(optimizer, step_size=self.scheduler_step, gamma=0.1)

            train_loss, val_loss, train_acc, val_acc = [], [], [], []

            for epoch in range(self.num_epochs):
                print(f"\n🔹 Epoch {epoch + 1}/{self.num_epochs}")

                train_epoch_loss, train_epoch_acc = self._run_epoch(model, optimizer, self.train_loader, training=True)
                train_loss.append(train_epoch_loss)
                train_acc.append(train_epoch_acc)

                val_epoch_loss, val_epoch_acc = self._run_epoch(model, optimizer, self.val_loader, training=False)
                val_loss.append(val_epoch_loss)
                val_acc.append(val_epoch_acc)

                scheduler.step()

                print(f"📌 Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc:.4f}")
                print(f"📌 Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}")

            # Test
            print("\n🚀 Running on Test Set...")
            test_loss, test_acc = self._run_epoch(model, optimizer, self.test_loader, training=False)
            print(f"\n✅ Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    def _run_epoch(self, model, optimizer, dataloader, training=True):
        if training:
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for data in tqdm(dataloader, desc="Training" if training else "Validating / Testing"):
            inputs = data["image"].to(self.device).float()
            labels = data["label"].to(self.device)

            if training:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = self.criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            outputs = torch.sigmoid(outputs)
            outputs = (outputs >= 0.5).float()

            correct += (outputs == labels).sum().item()
            total += labels.numel()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total

        return epoch_loss, epoch_acc


if __name__ == "__main__":
    from models import MLC_Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_models = [
    models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
    models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1),
    models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1),
    models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    ]

    models_list = [MLC_Model(8, base_model) for base_model in base_models]

    trainer = Train(
        models_list=models_list,
        csv_path="Variant-b(MultiLabel Classification)/Multi-Label dataset - with augmented.csv",
        num_epochs=10,
        criterion=nn.BCEWithLogitsLoss(),
        optimizer="SGD",
        scheduler=5,
        device=device,
        image_size=(224, 224),
        batch_size=1
    )

    trainer.train()