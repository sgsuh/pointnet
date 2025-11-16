"""
Create: 2022.07.07
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import torch
import argparse 

from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import PointSampler
from dataset import Normalize
from dataset import RandRotation_z
from dataset import ToTensor
from dataset import PointCloudData

from source.model import PointNet

def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    weights = [0.7226, 0.3229, 0.9666, 0.9877]
    class_weights = torch.FloatTensor(weights).cuda()

    criterion = torch.nn.NLLLoss(weight = class_weights)

    bs = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad = True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad = True).repeat(bs, 1, 1)

    if outputs.is_cuda:
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()

    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))

    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path",
                        type=str,
                        default="classification")
    parser.add_argument("--save_path",
                        type=str,
                        default="pointnet.pth")
    
    return parser.parse_args()

def main():
    args = make_parser()

    root_path = args.root_path

    train_transforms = transforms.Compose([PointSampler(128), Normalize(), RandRotation_z(), ToTensor()])

    train_ds = PointCloudData(root_path, transform = train_transforms)
    val_ds = PointCloudData(root_path, valid = True, folder = 'val')

    print('Train dataset size: ', len(train_ds))
    print('Valid dataset size: ', len(val_ds))

    train_loader = DataLoader(dataset = train_ds, batch_size = 16, shuffle = True)
    val_loader = DataLoader(dataset = val_ds, batch_size = 16)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = PointNet()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 10, verbose = True)

    min_loss = 1e6

    # Train
    for epoch in range(200):
        model.train()

        running_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs = data['pointcloud'].to(device).float()
            labels = data['category'].to(device)

            optimizer.zero_grad()

            outputs, m3x3, m64x64 = model(inputs.transpose(1, 2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print('[Epoch: {}, Batch {:04d}/{:04d}], loss: {:.3f}'.format(epoch + 1, i + 1, len(train_loader), running_loss / 10))

                running_loss = 0.0

        model.eval()

        correct = 0
        total = 0
        total_loss = 0.0

        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs = data['pointcloud'].to(device).float()
                    labels = data['category'].to(device)

                    outputs, m3x3, m64x64 = model(inputs.transpose(1, 2))
                    _, predicted = torch.max(outputs.data, 1)

                    val_loss = pointnetloss(outputs, labels, m3x3, m64x64)
                    total_loss += val_loss

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = 100. * correct / total
            total_loss /= len(val_loader)

            print('Val Acc: {:.3f}, Loss: {:.3f}'.format(val_acc, total_loss))

        scheduler.step(total_loss)

        if total_loss < min_loss:
            torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    main()