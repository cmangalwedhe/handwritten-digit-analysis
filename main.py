import numpy as np
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import torch, nn, optim

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])
train_set = datasets.MNIST('.\\Train Dataset', download=True, train=True, transform=transform)
value_set = datasets.MNIST('.\\Test Dataset', download=True, train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
value_loader = torch.utils.data.DataLoader(value_set, batch_size=64, shuffle=True)

data_iter = iter(train_loader)

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

nn_model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                         nn.ReLU(),
                         nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                         nn.ReLU(),
                         nn.Linear(hidden_sizes[1], output_size),
                         nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
images, labels = next(iter(train_loader))
print(images.dtype)
images = images.view(images.shape[0], -1)

logps = nn_model(images)
loss = criterion(logps, labels)

optimizer = optim.SGD(nn_model.parameters(), lr=0.003, momentum=0.9)
initial_time = time()

for i in range(15):
    running_loss = 0

    for images, labels in train_loader:
        images = images.view(images.shape[0], -1)

        optimizer.zero_grad()

        output = nn_model(images)
        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    else:
        print(f"Epoch {i + 1} - Training Loss: {running_loss / len(train_loader)}")

    print(f"Training Time: (in minutes) {(time() - initial_time) / 60}\n")

images, labels = next(iter(value_loader))
img = images[0].view(1, 784)

with torch.no_grad():
    logps = nn_model(img)


def view_classify(img, ps):
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


ps = torch.exp(logps)
probability = list(ps.numpy()[0])
print(f"Predicted Digit = {probability.index(max(probability))}")
view_classify(img.view(1, 28, 28), ps)

correct_count, count = 0, 0

for images, labels in value_loader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)

        with torch.no_grad():
            logps = nn_model(img)

        ps = torch.exp(logps)
        probability = list(ps.numpy()[0])
        predicted_label = probability.index(max(probability))
        true_label = labels.numpy()[i]

        if true_label == predicted_label:
            correct_count += 1

        count += 1

print(f"Number of Images Tested: {count}")
print(f"Model Accuracy: {(correct_count / count) * 100}%")

torch.save(nn_model, './my_mnist_model.pt')
