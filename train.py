import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, VGG16_BN_Weights, GoogLeNet_Weights
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import StepLR
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from data_processing import MultiLabelDataset, load_data
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np


class Train:

    def __init__(self, models_list: list, csv_path: str, num_epochs: int, criterion, optimizer: str,
                 scheduler: int, device, image_size_list: list, batch_size: int):

        assert len(models_list) == len(image_size_list), "🔴 ERROR: models_list and image_size_list must have the same number of elements!"

        self.models_list = models_list
        self.csv_path = csv_path
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.device = device
        self.image_size_list = image_size_list
        self.batch_size = batch_size

        train_df, val_df, test_df, class_names = load_data(self.csv_path)

        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self.train_loaders = []
        self.val_loaders = []
        self.test_loaders = []

        for img_size in self.image_size_list:
            train_dataset = MultiLabelDataset(train_df, img_size)
            val_dataset = MultiLabelDataset(val_df, img_size)
            test_dataset = MultiLabelDataset(test_df, img_size)

            self.train_datasets.append(train_dataset)
            self.val_datasets.append(val_dataset)
            self.test_datasets.append(test_dataset)

            self.train_loaders.append(DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2))
            self.val_loaders.append(DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2))
            self.test_loaders.append(DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2))

        if optimizer == "SGD":
            self.opt_func = lambda model: SGD(model.parameters(), lr=0.001, momentum=0.937)
        elif optimizer == "AdamW":
            self.opt_func = lambda model: AdamW(model.parameters(), lr=0.001)
        else:
            raise ValueError("Invalid optimizer, only 'SGD' or 'AdamW' is supported.")

        self.scheduler_step = scheduler

    def train(self):

        for idx, (base_model, img_size) in enumerate(zip(self.models_list, self.image_size_list)):
            model_name = type(base_model.model).__name__
            print(f"\n🔹 Training model: {model_name} | Image Size: {img_size}")

            model = base_model.to(self.device)
            optimizer = self.opt_func(model)
            scheduler = StepLR(optimizer, step_size=self.scheduler_step, gamma=0.1)

            train_loader = self.train_loaders[idx]
            val_loader = self.val_loaders[idx]
            test_loader = self.test_loaders[idx]

            train_loss, val_loss, train_acc, val_acc, train_f1, val_f1 = [], [], [], [], [], []

            best_val_loss = float("inf")

            model_dir = f"models/{model_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            os.makedirs(model_dir, exist_ok=True)

            best_model_path = os.path.join(model_dir, f"best_model_{model_name}.pth")
            csv_log_path = os.path.join(model_dir, f"training_log_{model_name}.csv")
            result_path = os.path.join(model_dir, f"results_{model_name}.png")

            for epoch in range(self.num_epochs):
                print(f"\n🔹 Epoch {epoch + 1}/{self.num_epochs}")

                train_epoch_loss, train_epoch_acc, train_epoch_f1 = self._run_epoch(model, optimizer, train_loader, training=True)
                train_loss.append(train_epoch_loss)
                train_acc.append(train_epoch_acc)
                train_f1.append(train_epoch_f1)


                val_epoch_loss, val_epoch_acc, val_epoch_f1 = self._run_epoch(model, optimizer, val_loader, training=False)
                val_loss.append(val_epoch_loss)
                val_acc.append(val_epoch_acc)
                val_f1.append(val_epoch_f1)

                scheduler.step()

                print(f"📌 Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc:.4f} | Train Macro F1: {val_epoch_f1:.4f}")
                print(f"📌 Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f} | Val Macro F1: {val_epoch_f1:.4f}")

                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    torch.save(model.state_dict(), best_model_path)
                    print(f"💾 Saved best model {model_name} with val_loss {val_epoch_loss:.4f}")

                df_log = pd.DataFrame({
                    "Epoch": list(range(1, len(train_loss) + 1)),
                    "Train Loss": train_loss,
                    "Train Acc": train_acc,
                    "Train F1": train_f1,
                    "Val Loss": val_loss,
                    "Val Acc": val_acc,
                    "Val F1": val_f1
                })
                df_log.to_csv(csv_log_path, index=False)

            plot_training_results(train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, model_name, result_path)

            print(f"\n🚀 Loading best model {model_name} and running on Test Set...")
            model.load_state_dict(torch.load(best_model_path))
            test_loss, test_acc, test_f1 = self._run_epoch(model, optimizer, test_loader, training=False)

            df_test_acc = pd.DataFrame({"Epoch": ["Test"],
                                        "Train Loss": [test_loss],
                                        "Train Acc": [test_acc],
                                        "Train F1": [test_f1],})
            df_test_acc.to_csv(csv_log_path, mode="a", index=False, header=False)

            print(f"\n✅ Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test Macro F1: {test_f1:.4f}")

    def _run_epoch(self, model, optimizer, dataloader, training=True):

        if training:
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        all_outputs = []
        all_labels = []
        num_batches = 0

        for data in tqdm(dataloader):

            inputs = data["image"].to(self.device).float()
            labels = data["label"].to(self.device).float()

            if training:
                optimizer.zero_grad()

            outputs = model(inputs)
            error = nn.BCEWithLogitsLoss()
            loss = error(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.cpu())

        epoch_loss = running_loss / len(dataloader)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        epoch_accuracy = self.calculate_accuracy(all_outputs, all_labels)
        epoch_macro_f1 = self.calculate_macro_f1(all_outputs, all_labels)

        return epoch_loss, epoch_accuracy, epoch_macro_f1


    def calculate_accuracy(self, outputs, labels):
        """
        Calculate accuracy based on model predictions.
        """
        preds = torch.sigmoid(outputs)
        preds = (preds >= 0.5).float()

        accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        return accuracy

    def calculate_macro_f1(self, outputs, labels):
        """
        Calculate Macro F1-score based on model prediction.
        """
        preds = torch.sigmoid(outputs)
        preds = (preds >= 0.5).float()

        macro_f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro", zero_division=0)
        return macro_f1


def plot_training_results(train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, model_name, result_path):

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label="Train Loss", marker='o')
    plt.plot(epochs, val_loss, label="Validation Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - {model_name}")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_acc, label="Train Accuracy", marker='o')
    plt.plot(epochs, val_acc, label="Validation Accuracy", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve - {model_name}")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_f1, label="Train F1", marker='o')
    plt.plot(epochs, val_f1, label="Validation F1", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Macro F1")
    plt.title(f"Macro F1 Curve - {model_name}")
    plt.legend()

    plt.tight_layout()

    filename = f"{result_path}"

    plt.savefig(filename, dpi=300)
    print(f"📸 Image has been saved: {filename}")

    plt.close()


if __name__ == "__main__":

    from models import MultiLabelClassification_Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_models = [
        models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1),
        # models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    ]

    models_list = [MultiLabelClassification_Model(8, base_model) for base_model in base_models]

    # Modify the image_size_list according to the model
    image_size_list = [(224, 224)]

    trainer = Train(
        models_list=models_list,
        csv_path="Variant-b(MultiLabel Classification)/Multi-Label dataset.csv",
        num_epochs=3,
        criterion=nn.BCEWithLogitsLoss(),
        optimizer="SGD",
        scheduler=5,
        device=device,
        image_size_list=image_size_list,
        batch_size=4
    )

    trainer.train()