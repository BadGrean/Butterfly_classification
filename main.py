import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models



# Define the neural network architecture
class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        self.resnet = torchvision.models.resnet18()
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

class CustomAlexNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomAlexNet, self).__init__()
        self.alexnet = torchvision.models.alexnet()
        in_features = self.alexnet.classifier[6].in_features
        self.alexnet.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.alexnet(x)


class CustomModelDensenet12(nn.Module):
    def __init__(self, num_classes):
        super(CustomModelDensenet12, self).__init__()
        self.base_model = torchvision.models.densenet121(pretrained=True)
        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def validate(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_write_file = 'custom_AlexNet_model.pth'

    num_classes = 100
    num_epochs = 0

    #model = CustomResNet(num_classes)
    model = CustomAlexNet(num_classes)
    #model = CustomVGG16(num_classes)
    #model = CustomModelDensenet12(num_classes)

    if input("write yes if you want to load trained model\n") == 'yes':
        saved_model_path = save_write_file
        state_dict = torch.load(saved_model_path)
        model.load_state_dict(state_dict)

    model.to(device)
    print(next(model.parameters()).device)
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0000001)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Define the relative path to your dataset
    train_path = 'train'
    val_path = 'valid'
    test_path = 'test'

    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)
    # Training loop

    for epoch in range(num_epochs):
        train(model, train_loader, optimizer, criterion, device)

        # Validation
        accuracy = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.2%}')

    # Test the model
    test_accuracy = validate(model, test_loader, criterion, device)
    with open('test_accuracy.txt', 'a') as file:
        # Append the print statement to the file
        print(f'Test Accuracy: {test_accuracy:.2%}', file=file)
    print(f'Test Accuracy: {test_accuracy:.2%}')
    if input("write yes if you want to save model\n") == 'yes':
    # Save the trained model
        torch.save(model.state_dict(), save_write_file)
