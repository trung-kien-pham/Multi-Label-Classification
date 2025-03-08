import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, VGG16_BN_Weights, Inception_V3_Weights, GoogLeNet_Weights
import torch.nn as nn
from torchvision import models
import torch

class MultiLabelClassification_Model(nn.Module):

    def __init__(self, num_classes: int, model: nn.Module) -> None:
        super(MLC_Model, self).__init__()
        
        self.model = model  # Save the model

        # fc in output
        if hasattr(self.model, "fc"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

        # classifier in output
        elif hasattr(self.model, "classifier"):
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, num_classes)

        # fc and AuxLogits in output
        elif hasattr(self.model, "AuxLogits"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

            # Change `AuxLogits` to support auxiliary loss as well
            in_features_aux = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)

        else:
            raise ValueError(f"No matching head found in model {type(self.model)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if hasattr(self.model, "AuxLogits") and self.training:
            # The models has 2 outputs when training.
            x, aux = self.model(x)
            return x, aux
            
        return self.model(x)  # Other models have 1 output


if __name__ == "__main__":

    Res_Net18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    print(Res_Net18.fc)
    print(Res_Net18.fc.in_features)
    print(Res_Net18.fc.out_features)

    model = MultiLabelClassification_Model(8, Res_Net18)

    print(model)
    print(model.model.fc)
    print(model.model.fc.in_features)
    print(model.model.fc.out_features)

    for names, parameters in model.named_parameters():

        print(f"{names}: {parameters.requires_grad}")

        if parameters.requires_grad:

            print(parameters.size())
            print(names, parameters.data)