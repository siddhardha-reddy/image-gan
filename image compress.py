import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import Encoder, Generator
import io

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model weights
netE = Encoder().to(device)
netG = Generator().to(device)
netE.load_state_dict(torch.load("netE.model", map_location=device))
netG.load_state_dict(torch.load("netG.model", map_location=device))
netE.eval()
netG.eval()

# Set up the Streamlit interface
st.title("Image Compression with Deep Learning")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Resize the original image to match the model dimensions
    original_size = image.size
    target_size = (218, 178)  # Use the dimensions from your model
    image = image.resize(target_size)

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

    # Display the original and reconstructed images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Original Image', use_column_width=True)
    with col2:
        st.image(reconstructed_img, caption='Reconstructed Image', use_column_width=True)

    # Add a download button for the reconstructed image
    buf = io.BytesIO()
    reconstructed_img.save(buf, format="JPEG")
    buf.seek(0)
    st.download_button(
        label="Download Reconstructed Image",
        data=buf,
        file_name="reconstructed_image.jpg",
        mime="image/jpeg"
    )