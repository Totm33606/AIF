import torch
import torchvision.transforms as transforms
from flask import Flask, send_file, request
from PIL import Image
import io
from model import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

MODEL_PATH = 'weights/unet.pth'

# Load the model
model = UNet()  # Assumes the UNet architecture is defined in model.py
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image
    transforms.Lambda(lambda img: img.convert('L') if img.mode == 'RGB' else img),  # Convert to grayscale if needed
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean=0.5 and std=0.5 for single-channel images
])

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was sent
    if not request.data:
        return "No image provided", 400
    else:
        img_binary = request.data

    # Open the image
    img = Image.open(io.BytesIO(img_binary))
    img_tensor = transform(img).unsqueeze(0).to(device)  # Apply transformations and add batch dimension

    # Make prediction with the model
    with torch.no_grad():
        outputs = model(img_tensor)  # Perform forward pass

    # Convert the prediction to a PIL Image
    pred_img_pil = transforms.ToPILImage()(outputs.squeeze(0).cpu())  # Remove batch dimension and convert to PIL Image

    # Save the image to a buffer
    buffer = io.BytesIO()
    pred_img_pil.save(buffer, format='PNG')
    buffer.seek(0)  # Move the cursor to the start of the stream

    return send_file(buffer, mimetype='image/png')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
