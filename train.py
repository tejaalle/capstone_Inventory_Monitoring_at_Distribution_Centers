# TODO: Import your dependencies.
# For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import OrderedDict
from PIL import ImageFile
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import logging

logging.basicConfig(level=logging.INFO,
                    format="'levelName:'%(levelname)s, 'ts':%(asctime)s, pathname: %(pathname)s, message: %(message)s")
logger = logging.getLogger("ModelLogger")


def test(model, test_loader, device, criterion):
    model.eval()
    running_loss = 0
    running_corrects = 0

    for inputs, target in test_loader:
        inputs = inputs.to(device)
        target = target.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, target)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == target.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)
    logger.info(f"Testing Accuracy: {100 * total_acc}, Testing Loss: {total_loss}")


def train(model, train_loader, valid_loader, criterion, optimizer, device, epochs):
    loaders = OrderedDict({'train': train_loader, 'valid': valid_loader})
    for epoch in range(epochs):
        # for each epoch do a training step and an evaluation step
        for load in loaders:
            total_loss = 0
            running_correct = 0

            if load == 'train':
                model.train()
            else:
                model.eval()

            for inputs, target in loaders[load]:
                inputs = inputs.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, target)

                _, preds = torch.max(outputs, 1)

                total_loss += loss.item() * inputs.size(0)
                with torch.no_grad():
                    running_correct += torch.sum(preds == target.data).item()

                if load == 'train':
                    # backward propagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            epoch_loss = total_loss / len(loaders[load].dataset)
            epoch_acc = running_correct / len(loaders[load].dataset)

            logger.info(f'for epoch= {epoch} load= {load}, loss and accuracy are epoch_loss={epoch_loss}, epoch_acc={epoch_acc}')

    logger.info("training done successfully")
    return model


def net():
    model = models.resnet50(pretrained=True)

    # Freezing convolution layers
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    # adding connected layers
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),  # layer
        nn.ReLU(inplace=True),  # activation function
        nn.Linear(256, 133))
    logger.info("model creation from pretrained model is success")

    return model


def create_data_loaders(data, batch_size):
    train_data_path = os.path.join(data, 'train_data')
    test_data_path = os.path.join(data, 'test_data')
    valid_data_path = os.path.join(data, 'valid_data')
    size = 224
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    logger.debug("creating transforms")
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    train_data = datasets.ImageFolder(train_data_path, train_transform)
    test_data = datasets.ImageFolder(test_data_path, test_transform)
    valid_data = datasets.ImageFolder(valid_data_path, test_transform)
    logger.info("loading data and tranforming..")
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    valid_data_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    logger.info("successfully loaded transformed data")

    return train_data_loader, test_data_loader, valid_data_loader


def main(args):
    model = net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device= {device}")
    model=model.to(device)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), args.lr)
    train_loader, test_loader, valid_loader = create_data_loaders(args.data_dir, args.batch_size)

    model = train(model, train_loader, valid_loader, loss_criterion, optimizer, device, args.epochs)
    test(model, test_loader, device, loss_criterion)

    path = os.path.join(args.model_dir, 'model.pth')
    logger.info(f"saving model to path={path}")
    torch.save(model, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size',
                        type=int,
                        default=64)
    parser.add_argument('--lr',
                        type=float,
                        default=0.1)
    parser.add_argument("--epochs",
                        type=int,
                        default = 10)
    parser.add_argument('--data_dir', type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir',
                        type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir',
                        type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])

    args = parser.parse_args()
    logger.info(f"parsed args are {args}")

    main(args)