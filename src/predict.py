import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from main import Network
import numpy as np
from scipy.spatial.distance import cdist



def load_model(weights_path):
    model = Network(1)
    state_dict = torch.load(weights_path)["model_state_dict"]
    
    # Adjust parameters to handle size mismatch
    state_dict["fc.2.weight"] = state_dict["fc.2.weight"][:1]
    state_dict["fc.2.bias"] = state_dict["fc.2.bias"][:1]
    
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image_path):

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path).convert("L")
    image = transform(image)

    return image.unsqueeze(0)

def predict_image(model, image, trained_embeddings_path):
    with torch.no_grad():
        output = model(image)

    # Load trained embeddings
    trained_embeddings = np.load(trained_embeddings_path)

    # Reshape output to match the shape of trained_embeddings
    output = output.repeat(trained_embeddings.shape[0], 1)

    # Compute Euclidean distances between output and trained embeddings
    distances = cdist(output.numpy(), trained_embeddings)

    # Find the index of the closest embedding
    closest_index = np.argmin(distances)

    # Get the label corresponding to the closest embedding
    closest_label = closest_index  # Replace this line with the correct label extraction method

    return closest_label

def main():
    weights_path = "trained_model.pth"
    trained_embeddings_path = "trained_embeddings.npy"
    model = load_model(weights_path)

    image_path = 'uploaded_image.png'
    image = preprocess_image(image_path)

    closest_label = predict_image(model, image, trained_embeddings_path)
    print("Closest Label:", closest_label)

if __name__ == "__main__":
    main()
