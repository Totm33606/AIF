from pathlib import Path
import shutil

import argparse
from statistics import mean

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from model import MNISTNet

# Définir l'appareil (GPU si disponible, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(net, optimizer, loader, epochs, writer=None):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}')

            if writer:
                writer.add_scalar('training loss', mean(running_loss), epoch)

def test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total
	

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='MNIST', help='experiment name')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training and testing')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--save_model', type=str, default='mnist_net.pth', help='path to save the trained model')

    args = parser.parse_args()
    exp_name = args.exp_name
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    save_model_path = args.save_model

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Datasets
    trainset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
    testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)

    # Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Instanciation du modèle et de l'optimiseur
    net = MNISTNet()
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)

    # Writer for tensorboard
    path_runs = Path("runs")
    if not path_runs.exists():
        path_runs.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(f'runs/MNIST')

    # Entraînement
    train(net, optimizer, trainloader, epochs, writer)

    # Évaluation du modèle
    net.eval()
    test_acc = test(net, testloader)
    print(f'Test accuracy: {test_acc}%')

    # Sauvegarde du modèle
    path_weigts = Path("weights")
    if not path_weigts.exists(): 
        path_weigts.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), path_weigts / save_model_path)
    print(f'Model saved to {path_weigts / save_model_path}')