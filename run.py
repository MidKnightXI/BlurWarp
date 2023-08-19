import torch
import torch.backends.mps
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import nn
import json

DEVICE = torch.device("cpu")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

print(f"Running on {DEVICE}")

class BlurDetectionModel(nn.Module):
    def __init__(self):
        super(BlurDetectionModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 64 * 64, 256)  # Adjust based on your input size
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten each image in the batch
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Load the trained model
model = BlurDetectionModel()
model.load_state_dict(torch.load('blur_detection_model.tch'))
model.to(DEVICE)

# Set up data transformations for inference
transform = transforms.Compose([
    transforms.Resize((256, 256), antialias=True),
    transforms.ToTensor(),
])

# Create a DataLoader for inference
inference_dataset = datasets.ImageFolder(root='dataset', transform=transform)
inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

# Run inference
model.eval()  # Set the model to evaluation mode
results = []

predictions = []

with torch.no_grad():
    for batch_idx, (data, entry) in enumerate(zip(inference_loader, inference_dataset.samples)):
        data_resized = data[0].to(DEVICE)
        output = model(data_resized).item()

        prediction = {
            "status": "success",
            "filename": entry[0],
            "detected": {"blur_score": output}
        }
        predictions.append(prediction)


# Save the predictions as a JSON file
output_file = "inference_results.json"
with open(output_file, "w") as f:
    json.dump(predictions, f, indent=4)

print(f"Results saved to {output_file}")
