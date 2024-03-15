from flask import Flask, render_template, request, send_from_directory, send_file
import base64
from PIL import Image
import torch
from torchvision import transforms
from model import Encoder, Generator
import io
import os

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model weights
netE = Encoder().to(device)
netG = Generator().to(device)
netE.load_state_dict(torch.load("netE.model", map_location=device))
netG.load_state_dict(torch.load("netG.model", map_location=device))
netE.eval()
netG.eval()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/compressImage', methods=['POST'])
def index():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    uploaded_file = request.files['image']

    if uploaded_file.filename == '':
        return 'No image selected', 400

    # Save the original image
    original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_image.jpg')
    uploaded_file.save(original_image_path)

    original_image = Image.open(uploaded_file).convert('RGB')
    original_size = original_image.size

    # Resize the original image to match the model dimensions
    target_size = (218, 178)  # Use the dimensions from your model
    image = original_image.resize(target_size)

    # Preprocess the image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Compress and reconstruct the image
    with torch.no_grad():
        encoded_img = netE(image_tensor)
        reconstructed_img = netG(encoded_img).cpu()

    # Postprocess the reconstructed image
    reconstructed_img = (reconstructed_img.squeeze() * 0.5 + 0.5).clamp(0, 1)
    reconstructed_img = transforms.ToPILImage()(reconstructed_img)

    # Resize the compressed image to match the original dimensions
    compressed_image = reconstructed_img.resize(original_size)

    # Create a BytesIO object to store the compressed image data
    compressed_image_bytes = io.BytesIO()
    compressed_image.save(compressed_image_bytes, format='JPEG')
    compressed_image_bytes.seek(0)

    # Convert the compressed image to base64
    compressed_image_data = base64.b64encode(compressed_image_bytes.getvalue()).decode('utf-8')

    # Generate the download URL for the compressed image
    compressed_image_download_url = '/download_compressed_image'

    return render_template('index.html', compressed_image_data=compressed_image_data, compressed_image_download_url=compressed_image_download_url, compressed_image_bytes=compressed_image_bytes)

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/download_compressed_image')
def download_compressed_image(compressed_image_bytes=None):
    if compressed_image_bytes is None:
        return 'No compressed image available', 404

    compressed_image_bytes.seek(0)
    return send_file(compressed_image_bytes, mimetype='image/jpeg', as_attachment=True, download_name='compressed_image.jpeg')

if __name__ == '__main__':
    app.run(debug=True)