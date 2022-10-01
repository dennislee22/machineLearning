import torch
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', 
                                 download=True, 
                                 train=True, 
                                 transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=64, 
                                          shuffle=True,num_workers=3,pin_memory=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', 
                                download=True, 
                                train=False, 
                                transform=transform)
testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=64, 
                                         shuffle=True,num_workers=3,pin_memory=True)

import matplotlib.pyplot as plt
import numpy as np

def imshow(image, ax=None, title=None, normalize=True):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax
  
image, label = next(iter(trainloader))
imshow(image[0,:]);

from torch import nn

model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

device = torch.device("cuda")
model.to(device)
from torch import optim

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=0.003)


epoch = 1

%%time
#while True:
for epoch in range(1, 10):
    running_loss = 0
    model.train()
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images, labels = images.to(device), labels.to(device)
        images = images.view(images.shape[0], -1)
        
        optimizer.zero_grad()

        output = model(images)
  
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    training_loss = running_loss/len(trainloader)
    print("Epoch, Loss:    {:2}, {:1.3}".format(epoch, training_loss))
    epoch += 1
    
    #if training_loss < 0.4:
    #    break
        
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

def view_classify(img, ps):
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(['T-shirt/top',
                        'Trouser',
                        'Pullover',
                        'Dress',
                        'Coat',
                        'Sandal',
                        'Shirt',
                        'Sneaker',
                        'Bag',
                        'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

#print("Models layer keys: \n\n", model.state_dict().keys())
#torch.save(model.state_dict(), 'model.pth')
#state_dict = torch.load('model.pth')
#print(state_dict.keys())
#model.load_state_dict(state_dict)

def main():
  dataiter = iter(testloader)
  for _ in range(10):
    device = torch.device("cpu")
    images, labels = dataiter.next()
    img = images[0]

    # Convert 2D image to 1D vector
    img = img.resize_(1, 784)

    # Turn off gradients to speed up this part
    with torch.no_grad():
        model.to(device)
        logps = model(img)
    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)

    # Plot the image and probabilities
    view_classify(img.resize_(1, 28, 28), ps)
    
if __name__ == "__main__":  
    main()
