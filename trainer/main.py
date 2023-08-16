import torch
import torch.backends.mps
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import transforms
from torchvision import datasets

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


transform = transforms.Compose([
    transforms.Resize((256, 256), antialias=True),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root='./dataset', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
model = BlurDetectionModel()
model.to(DEVICE)  # Move model to the mps device if it's available
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)  # Learning rate scheduler

# Training loop
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        # Resize images to 256x256 before feeding into the model
        data_resized = torch.cat([transforms.Resize((256, 256), antialias=True)(img.unsqueeze(0)) for img in data], dim=0)

        data_resized = data_resized.to(DEVICE)  # Move data to the mps device if it's available
        target = target.to(DEVICE)  # Move target to the mps device if it's available

        output = model(data_resized)
        loss = criterion(output, target.float().view_as(output))
        loss.backward()

        optimizer.step()
        scheduler.step()

        pred = output.round()
        correct = pred.eq(target.view_as(pred)).sum().item()
        precision = correct / len(target)

        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}, Precision: {precision:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'blur_detection_model.tch')
