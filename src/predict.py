import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from main import Network


def load_model(weights_path):
    model = Network(1)
    model.load_state_dict(torch.load(weights_path)["model_state_dict"])
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

def predict_image(model, image):

    with torch.no_grad():
        output = model(image)
    return output

def main():
   
    weights_path = "trained_model.pth"
    model = load_model(weights_path)
    



    image_path = 'uploaded_image.png'
   
    image = preprocess_image(image_path)
    



    output = predict_image(model, image)
    print("Output:", output)

if __name__ == "__main__":
    main()
