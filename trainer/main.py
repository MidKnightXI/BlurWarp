import pandas as pd
import torch
import torch.backends.mps
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets.folder import default_loader


def setup_device():
    DEVICE = torch.device("cpu")

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")

    print(f"Running on {DEVICE}")
    return DEVICE


class BlurDetectionModel(nn.Module):
    def __init__(self):
        super(BlurDetectionModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 64 * 64, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class BlurrySharpDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, transform=None):
        self.data = annotations
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        image = default_loader(img_path)
        if self.transform:
            image = self.transform(image)

        return image, label


def preprocess_target_values(target):
    return torch.tensor([1 if label == 'blurry' else 0 for label in target])


def train_blur_detection_model(annotations_path, output_model_path, num_epochs=70):
    DEVICE = setup_device()

    annotations = pd.read_csv(annotations_path)

    transform = transforms.Compose([
        transforms.Resize((256, 256), antialias=True),
        transforms.ToTensor(),
    ])

    train_dataset = BlurrySharpDataset(annotations, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = BlurDetectionModel()
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            data = data.to(DEVICE)
            label_batch = preprocess_target_values(target).to(DEVICE)

            output = model(data)
            loss = criterion(output, label_batch.float().view_as(output))
            loss.backward()

            optimizer.step()

            pred = output.round()
            correct = pred.eq(label_batch.view_as(pred)).sum().item()
            precision = correct / len(label_batch)

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Precision: {precision:.4f}')

        scheduler.step()

    torch.save(model.state_dict(), output_model_path)
    print(f'Model saved to {output_model_path}')


if __name__ == '__main__':
    annotations_path = './dataset/annotations.csv'
    output_model_path = '../blur_detection_model.tch'
    train_blur_detection_model(annotations_path, output_model_path)
