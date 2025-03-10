# from sklearn.metrics import accuracy_score, f1_score

# y_pred = [[0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1]]
# y_true = [[0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1]]

# print("Accuracy: ", accuracy_score(y_true, y_pred))
# print("F1 score: ", f1_score(y_true, y_pred, average="macro"))

import torch
import torch.nn as nn
from models import MultiLabelClassification_Model
from torchvision import models
from torchvision.models import ResNet18_Weights, GoogLeNet_Weights, VGG16_BN_Weights
import torchvision.transforms as transforms
from PIL import Image

model = MultiLabelClassification_Model(8, models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1))

state_dict = torch.load('models/GoogLeNet_2025-03-09_20-11-14/best_model_GoogLeNet.pth')

model.load_state_dict(state_dict)

# print(model)
print(model.model.fc)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open("Variant-b(MultiLabel Classification)/test/IMG20220323094543_3.jpg")
input_data = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_data)

preds = torch.sigmoid(output)
preds = (preds >= 0.5).float()

print(preds)