from argparse import ArgumentParser
from os import listdir
from os.path import isdir, join, dirname, abspath
from sys import exit as sys_exit, stdout
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision.models as models
from PIL import Image, UnidentifiedImageError
import json


DEVICE = torch.device("cpu")


class BlurDetectionResNet(nn.Module):
    def __init__(self):
        super(BlurDetectionResNet, self).__init__()
        self.resnet = models.resnet50()
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x) -> torch.Tensor:
        x = self.resnet(x)
        return torch.sigmoid(x)


class SingleFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for image_file in listdir(root_dir):
            img_path = join(root_dir, image_file)
            try:
                Image.open(img_path)
                self.samples.append((image_file, None))
            except UnidentifiedImageError:
                stdout.write(f"Cannot identify image file '{img_path}', skipping.\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = join(self.root_dir, self.samples[idx][0])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


def setup_argparse() -> ArgumentParser:
    default_output_path = join(dirname(abspath(__file__)), "predictions.json")

    parser = ArgumentParser(
        prog="blurwarp",
        description="Detection of blurry images using ResNet50 AI model",
        epilog="If you encounter any problem please submit an issue here: https://github.com/MidKnightXI/BlurWarp")

    parser.add_argument("-t", "--target",
                        type=str,
                        required=True,
                        help="Define in which directory the model will analyze the images")
    parser.add_argument("-o", "--output",
                        default=default_output_path,
                        type=str,
                        help="Define the path of the output file eg: ./out/pred.json")
    args = parser.parse_args()
    return args


def setup_model() -> BlurDetectionResNet:
    model_path = join(dirname(abspath(__file__)), "blur_detection_model.tch")
    model = BlurDetectionResNet()
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()
    stdout.write("Model loaded\n")
    return model


def dump_predictions(path: str, predictions: list) -> None:
    with open(path, "w") as f:
        json.dump(predictions, f, indent=2)
    stdout.write(f"Results saved to {path}\n")


def run_model(path: str, output_path: str) -> None:
    model = setup_model()

    transform = transforms.Compose([
        transforms.Resize((256, 256), antialias=True),
        transforms.ToTensor(),
    ])

    # todo: add recursivity as an argument and stop loading directories and not images
    dataset = SingleFolderDataset(root_dir=path, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    z_loader_dataset = zip(loader, dataset.samples)

    predictions = list()

    with torch.no_grad():
        for _, (data, entry) in enumerate(z_loader_dataset):
            if data is None:
                continue

            data = data[0].unsqueeze(0).to(DEVICE)
            output = model(data).item()

            predictions.append({
                "status": True if round(output, 2) > 0.9 else False,
                "filename": entry[0],
                "score": round(output, 2)
            })

    dump_predictions(output_path, predictions)


if __name__ == "__main__":
    args = setup_argparse()

    if isdir(args.target) == False:
        stdout.write("Please specify a proper path: path/to/directory\n")
        sys_exit(1)

    stdout.write(f"Using - {DEVICE} - backend to run the model\n")
    run_model(args.target, args.output)