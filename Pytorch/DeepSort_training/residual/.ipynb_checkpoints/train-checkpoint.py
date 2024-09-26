import os
import time
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
from dataset import CustomDataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import pickle

from torch.utils.data import random_split

from cspDensenet_modules import csp_densenet121
from model import Net

# Paramètres
data_dir = '/home/jovyan/Desktop/deep_sort_residual_1720'

lr = 1e-3
interval = 500
device = "cuda:0" if torch.cuda.is_available() else "cpu"
test_size = 0.4


# Définition des transformations pour l'ensemble d'entraînement
#mean = [0.485, 0.456, 0.406]
#std = [0.229, 0.224, 0.225]

mean = [0, 0, 0]
std = [1,1,1]

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(64),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)  
])

test_transforms = transforms.Compose([
    # transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean, std) 
])

# Chargement des données
dataset = CustomDataset(data_dir)

train_len = int(len(dataset)*(1-test_size))  # calculer la longueur de l'ensemble d'apprentissage
test_len = len(dataset) - train_len  # calculer la longueur de l'ensemble de test
train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

train_dataset.dataset.set_transform(train_transforms)
test_dataset.dataset.set_transform(test_transforms)

trainloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

print(f"Nombre total d'échantillons: {len(dataset)}")
print(f"Nombre d'échantillons d'entrainement: {len(train_dataset)}")
print(f"Nombre d'échantillons de test: {len(test_dataset)}")

# Définition du réseau
#net = csp_densenet121(num_classes=785)  
net = Net(num_classes=1200)  
net.to(device)

# Fonction de perte et optimiseur
criterion = CrossEntropyLoss()
optimizer = AdamW(net.parameters(), lr)

# Train function for each epoch
def train(epoch):
    print("\nEpoch : %d" % (epoch + 1))
    net.train()
    cumulative_loss = 0.  # Variable for accumulating the loss over all batches
    interval_loss = 0.  # Initialize interval loss for periodic display
    correct = 0
    total = 0
    start = time.time()

    for idx, (inputs, labels) in enumerate(trainloader):
        # Forward
        labels = labels - 1
        assert labels.max() < 1200, f"Label {labels.max()} out of bounds."
        assert labels.min() >= 0, f"Label {labels.min()} out of bounds."

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
        optimizer.step()

        # Accumulating
        cumulative_loss += loss.item()  # Add loss for this batch to the cumulative loss
        interval_loss += loss.item()    # Add loss for this batch to the interval loss
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # Print 
        if (idx + 1) % interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Interval Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100. * (idx + 1) / len(trainloader), end - start, interval_loss / interval, correct, total, 100. * correct / total
            ))
            interval_loss = 0.  # Reset interval loss after display
            start = time.time()
    return cumulative_loss / len(trainloader), 100. * correct / total  # Return the average loss and accuracy


def test(epoch):
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            labels = labels - 1
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)
        
        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(testloader), end-start, test_loss/len(testloader), correct, total, 100.*correct/total
            ))
    return test_loss / len(testloader), 100. * correct / total


def adjust_learning_rate(optimizer, lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1)
    print('adjusting lr to',lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main(lr):
    
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    
    model_dir = './weights_ds_3'

    start_epoch = 0
    num_epochs = 40
    #resume_path = 'weights_ds_3/model_epoch_17_acc_68.41_loss_1.159.pth'
    resume_path = ''
    if resume_path:
        print(f'Loading checkpoint from {resume_path}')
        checkpoint = torch.load(resume_path)
        net.load_state_dict(checkpoint['net_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        # Save the loss and accuracy lists after each epoch
        with open('results/resnet/train_loss_list.pkl', 'wb') as f:
            pickle.dump(train_loss_list, f)
        with open('results/resnet/train_acc_list.pkl', 'wb') as f:
            pickle.dump(train_acc_list, f)
        with open('results/resnet/test_loss_list.pkl', 'wb') as f:
            pickle.dump(test_loss_list, f)
        with open('results/resnet/test_acc_list.pkl', 'wb') as f:
            pickle.dump(test_acc_list, f)
        
        
        print('learning rate:', lr)
        if epoch % 15 == 0 and epoch != 0:
            lr = adjust_learning_rate(optimizer, lr, epoch)
        if epoch > 20:
            torch.save({
                'epoch': epoch,
                'net_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                }, os.path.join(model_dir, f'model_epoch_{epoch}_acc_{test_acc:.2f}_loss_{test_loss:.3f}.pth'), _use_new_zipfile_serialization=False)

        
        # Plotting Loss
    plt.figure()
    plt.plot(range(num_epochs), train_loss_list, 'r', label='Training Loss')
    plt.plot(range(num_epochs), test_loss_list, 'b', label='Testing Loss')
    plt.title('Loss over 40 epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/resnet/loss.png')
    plt.show()
    
    # Plotting Accuracy
    plt.figure()
    plt.plot(range(num_epochs), train_acc_list, 'r', label='Training Accuracy')
    plt.plot(range(num_epochs), test_acc_list, 'b', label='Testing Accuracy')
    plt.title('Accuracy over 40 epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('results/resnet/Accuracy.png')
    plt.show()
    return lr

if __name__ == '__main__':
    lr = main(lr)
