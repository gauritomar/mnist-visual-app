import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from main import Network

def get_embeddings(model, data_loader):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, targets in data_loader:
            output = model(images)
            embeddings.append(output.numpy())
            labels.append(targets.numpy())
        
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    return embeddings, labels


def main():
    embedding_dims = 2
    model = Network(embedding_dims)
    checkpoint = torch.load("trained_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
    trained_embeddings, trained_labels = get_embeddings(model, train_loader)

    np.save("trained_embeddings.npy", trained_embeddings)
    np.save("trained_labels.npy", trained_labels)

if __name__ == "__main__":
    main()