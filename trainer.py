import pandas as pd
import torch
import torch.backends.mps
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import torchvision.models as models

def setup_device():
    DEVICE = torch.device("cpu")

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")

    print(f"Running on {DEVICE}")
    return DEVICE


class BlurDetectionResNet(nn.Module):
    def __init__(self):
        super(BlurDetectionResNet, self).__init__()
        self.resnet = models.resnet18()
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = self.resnet(x)
        return torch.sigmoid(x)



class BlurrySharpDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, transform=None):
        self.data = annotations
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label_str = self.data.iloc[idx, 1]

        image = default_loader(img_path)
        if self.transform:
            image = self.transform(image)

        label = 1 if label_str == 'blurry' else 0
        return image, label


def train_blur_detection_model(annotations_path, output_model_path):
    EPOCH = 10
    DEVICE = setup_device()

    annotations = pd.read_csv(annotations_path)

    transform = transforms.Compose([
        transforms.Resize((256, 256), antialias=True),
        transforms.ToTensor(),
    ])

    train_dataset = BlurrySharpDataset(annotations, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = BlurDetectionResNet()
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(EPOCH):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            data = data.to(DEVICE)
            label_batch = target.to(DEVICE).view(-1, 1)  # Reshape target labels to match model output shape

            output = model(data)
            loss = criterion(output, label_batch.float())
            loss.backward()

            optimizer.step()

            pred = (output >= 0.5).float()  # Round predictions for accuracy calculation
            correct = pred.eq(label_batch).sum().item()
            precision = correct / len(label_batch)

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{EPOCH}], Loss: {loss.item():.4f}, Precision: {precision:.4f}')

        scheduler.step()

    torch.save(model.state_dict(), output_model_path)
    print(f'Model saved to {output_model_path}')


if __name__ == '__main__':
    annotations_path = 'dataset/annotations.csv'
    output_model_path = 'blur_detection_model.tch'
    train_blur_detection_model(annotations_path, output_model_path)
