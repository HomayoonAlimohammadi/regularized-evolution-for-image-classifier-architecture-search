import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ModelModule(nn.Module):
    def __init__(self, architecture):
        super(ModelModule, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = 1  # example for MNIST dataset

        for layer in architecture:
            layer_type, layer_params = layer.split('-')
            if layer_type == 'Conv':
                n_out = int(layer_params)
                self.layers.append(nn.Conv2d(in_channels, n_out, kernel_size=3, padding=1))
                self.layers.append(nn.ReLU())
                in_channels = n_out
            elif layer_type == 'MaxPool':
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif layer_type == 'AvgPool':
                self.layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            elif layer_type == 'Dense':
                n_out = int(layer_params)
                self.layers.append(nn.Linear(in_channels, n_out))
                self.layers.append(nn.ReLU())
                in_channels = n_out

        # Add a final linear layer based on the number of classes (e.g., 10 for MNIST)
        self.layers.append(nn.Linear(in_channels, 10))

        self.layers.append(nn.Softmax(10))

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = x.view(x.size(0), -1)
            x = layer(x)
        return x

def train_and_evaluate(architecture, n_epochs):
    # initialize 
    model = ModelModule(architecture)
    model.train()

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # load mnist
    train_dataset = datasets.MNIST('.', train=True, download=True,
                                   transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # training loop
    for _ in range(n_epochs):  # example with 2 epochs
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # evaluate
    model.eval()
    test_dataset = datasets.MNIST('.', train=False, download=True,
                                  transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)

    return accuracy

class Model:
    def __init__(self, architecture, accuracy=0):
        self.architecture = architecture
        self.accuracy = accuracy
        self.age = 0

def random_architecture():
    # possible layers and their parameters
    layer_types = ['Conv', 'MaxPool', 'AvgPool', 'Dense']
    num_layers = random.randint(2, 5)  # Randomly choose the number of layers

    architecture = []
    for _ in range(num_layers):
        layer_type = random.choice(layer_types)

        if layer_type in ['Conv', 'Dense']:
            # for Conv and Dense layers, choose a random number of units/filters
            units_or_filters = random.choice([16, 32, 64, 128])
            architecture.append(f"{layer_type}-{units_or_filters}")

        else:
            # for pooling layers, we won't specify units/filters
            architecture.append(layer_type)

    return architecture


def mutate(architecture):
    # Placeholder for mutating the architecture
    return architecture + "_mutated"

def regularized_evolution(n_iters, population_size, sample_size, n_epochs):
    population = []
    history = []

    # Initialize population
    while len(population) < population_size:
        architecture = random_architecture()
        accuracy = train_and_evaluate(architecture, n_epochs)
        model = Model(architecture, accuracy)
        population.append(model)
        history.append(model)

    # Evolution
    for _ in range(cycles):
        sample = random.sample(population, sample_size)
        parent = max(sample, key=lambda i: i.accuracy)
        child_arch = mutate(parent.architecture)
        child_accuracy = train_and_evaluate(child_arch)
        child = Model(child_arch, child_accuracy)

        population.append(child)
        history.append(child)
        population.sort(key=lambda i: i.age)
        population.pop(0)  # Remove the oldest model

        # Increment age of each model
        for model in population:
            model.age += 1

    return max(history, key=lambda i: i.accuracy)

def main():
    print(random_architecture())
    # best_model = regularized_evolution(
    #     cycles=100, 
    #     population_size=50, 
    #     sample_size=10
    # )
    # print("Best model architecture:", best_model.architecture)
    # print("Best model accuracy:", best_model.accuracy)


if __name__ == "__main__":
    main()
