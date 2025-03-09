import torch.nn as nn
from torchvision import models
from torchvision.models import Inception_V3_Weights
import torch

class MultiLabelClassification_Model(nn.Module):

    def __init__(self, num_classes: int, model: nn.Module) -> None:
        super(MultiLabelClassification_Model, self).__init__()
        
        self.model = model

        if hasattr(self.model, "fc"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

        elif hasattr(self.model, "classifier"):
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, num_classes)

        elif hasattr(self.model, "AuxLogits"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

            in_features_aux = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(in_features_aux, num_classes) #Sửa lại thành num_classes

        else:
            raise ValueError(f"No matching head found in model {type(self.model)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if hasattr(self.model, "AuxLogits") and self.training:
            x, aux = self.model(x)
            return x, aux
            
        return self.model(x)

if __name__ == "__main__":

    Inception_V3 = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

    print("Original fc:", Inception_V3.fc)
    print("Original AuxLogits.fc:", Inception_V3.AuxLogits.fc)

    model = MultiLabelClassification_Model(8, Inception_V3)

    print("Modified fc:", model.model.fc)
    print("Modified AuxLogits.fc:", model.model.AuxLogits.fc)