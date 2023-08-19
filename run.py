import torch
import torch.backends.mps
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import nn
from torchvision import transforms
from torchvision import datasets
import torchvision.models as models
import json

DEVICE = torch.device("cpu")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

print(f"Running on {DEVICE}")

class BlurDetectionResNet(nn.Module):
    def __init__(self):
        super(BlurDetectionResNet, self).__init__()
        self.resnet = models.resnet18()
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = self.resnet(x)
        return torch.sigmoid(x)

model = BlurDetectionResNet()
model.load_state_dict(torch.load('blur_detection_model.tch'))
model.to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((256, 256), antialias=True),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root='dataset', transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

model.eval()
results = []

predictions = []

with torch.no_grad():
    for batch_idx, (data, entry) in enumerate(zip(loader, dataset.samples)):
        data_resized = data[0].to(DEVICE)
        output = model(data_resized).item()

        prediction = {
            "status": "success",
            "filename": entry[0],
            "detected": {"blur_score": output}
        }
        predictions.append(prediction)


output_file = "predictions.json"
with open(output_file, "w") as f:
    json.dump(predictions, f, indent=4)

print(f"Results saved to {output_file}")
