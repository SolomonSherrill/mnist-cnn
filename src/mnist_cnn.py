import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2
torch.backends.nnpack.enabled = False
train_data = datasets.MNIST(
    root="/Users/solomon/mnist-cnn/data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.MNIST(
    root="/Users/solomon/mnist-cnn/data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data,batch_size=batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(1,32,(5,5),padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,(5,5),padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7,10)
        )
    def forward(self,x):
        head_input = self.cnn_stack(x)
        logits = self.head(head_input)
        return logits

model = NeuralNetwork()
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
model = NeuralNetwork().to(device)

lr = 1e-3
epochs = 7
optimizer = torch.optim.Adam(model.parameters(),lr = lr)
loss_function = nn.CrossEntropyLoss()

def train_loop(dataloader,model,loss_function,optimizer,epoch):
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_function(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch} complete")

def test_loop(dataloader,model):
    model.eval()
    set_size = len(dataloader.dataset)
    correct = 0
    with torch.no_grad():
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    print(f"Average accuracy: {(correct/set_size)*100}%\n\n")

def save_model(model,path):
    torch.save(model.state_dict(), path)

path = "/Users/solomon/mnist-cnn/cnn_weights.pt"
for i in range(epochs):
    train_loop(train_loader,model,loss_function,optimizer,i+1)
    test_loop(test_loader,model)
save_model(model,path)
