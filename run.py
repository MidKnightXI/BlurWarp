from argparse import ArgumentParser
from os.path import isdir
from sys import exit as sys_exit
import torch
import torch.backends.mps
from torch.utils.data import DataLoader
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


class BlurDetectionResNet(nn.Module):
    def __init__(self):
        super(BlurDetectionResNet, self).__init__()
        self.resnet = models.resnet50()
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x) -> torch.Tensor:
        x = self.resnet(x)
        return torch.sigmoid(x)


def setup_argparse() -> ArgumentParser:
    parser = ArgumentParser(
        prog="blurwarp",
        description="Detection of blurry images using ResNet50 AI model",
        epilog="If you encounter any problem please submit an issue here: https://github.com/MidKnightXI/BlurWarp")

    parser.add_argument("-d", "--directory",
                        type=str,
                        required=True,
                        help="Define in which directory the model will analyze the images")
    parser.add_argument("-o", "--output",
                        default="predictions.json",
                        type=str,
                        help="Define the path of the output file eg: ./out/pred.json")
    args = parser.parse_args()
    return args


def setup_model() -> BlurDetectionResNet:
    model = BlurDetectionResNet()
    model.load_state_dict(torch.load('blur_detection_model.tch'))
    model.to(DEVICE)
    model.eval()
    print("Model loaded")
    return model


def dump_predictions(path: str, predictions: list) -> None:
    with open(path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Results saved to {path}")


def run_model(path: str, output_path: str) -> None:
    model = setup_model()

    transform = transforms.Compose([
        transforms.Resize((256, 256), antialias=True),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=path, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    z_loader_dataset = zip(loader, dataset.samples)

    predictions = list()

    with torch.no_grad():
        for _, (data, entry) in enumerate(z_loader_dataset):
            output = model(data[0].to(DEVICE)).item()

            predictions.append({
                "status": "success",
                "filename": entry[0],
                "detected": {"blur_score": round(output, 2)}
            })

    dump_predictions(output_path, predictions)


if __name__ == "__main__":
    args = setup_argparse()

    if isdir(args.directory) == False:
        print("Please specify a proper path: path/to/directory")
        sys_exit(1)

    print(f"Using - {DEVICE} - backend to run the model")
    run_model(args.directory, args.output)