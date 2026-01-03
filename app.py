# Version 2 (Nice version)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float

st.title("📸 Image Compression using SVD (Singular Value Decomposition)")

st.write("""
This app compresses an image using **SVD**, a mathematical technique that keeps only the most important parts
of an image, reducing file size while keeping visual quality.
""")

st.markdown("""
### 🧠 How this works:
1. The image is represented as a **matrix of numbers** (pixels).  
2. **SVD** breaks it into three smaller matrices: **U**, **S**, and **Vᵗ**.  
3. We keep only the **top k** singular values — these represent the most important image details.  
4. A **smaller k** gives higher compression (but lower image quality).  
   A **larger k** gives better quality (but less compression).  

📘 **In short:**  
Mathematics helps us find what parts of the image matter most and throw away the rest — just like summarizing a big story into a few key lines.
""")

# --- Upload Image ---
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = img_as_float(io.imread(uploaded_file))

    # --- Handle RGBA or Grayscale ---
    if image.ndim == 2:  # grayscale
        st.info("Detected grayscale image.")
        image_rgb = np.stack((image,)*3, axis=-1)
    elif image.shape[2] == 4:  # RGBA
        st.info("Detected RGBA image (with transparency). Converting to RGB.")
        image_rgb = image[..., :3]
    else:
        image_rgb = image

    st.image(image_rgb, caption="Original Image", use_container_width=True)

    # --- Choose compression level ---
    k = st.slider("Select number of singular values to keep (compression level)", 5, 200, 50)

    # --- Apply SVD to each color channel ---
    compressed_channels = []
    for i in range(3):  # RGB
        U, S, VT = np.linalg.svd(image_rgb[:, :, i], full_matrices=False)
        S[k:] = 0  # Keep only top-k singular values
        compressed_channel = np.dot(U, np.dot(np.diag(S), VT))
        compressed_channels.append(compressed_channel)

    compressed_image = np.stack(compressed_channels, axis=2)

    # --- Display results ---
    st.subheader("Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="Original", use_container_width=True)
    with col2:
        st.image(np.clip(compressed_image, 0, 1), caption=f"Compressed (k={k})", use_container_width=True)

    # --- Compute compression ratio ---
    orig_size = np.prod(image_rgb.shape)
    compressed_size = (U.shape[0]*k + k + k*VT.shape[0]) * 3  # Approx. per channel
    compression_ratio = compressed_size / orig_size
    st.write(f"*Compression Ratio:* {compression_ratio:.2%} (smaller is better)")

else:
    st.info("👆 Upload an image to begin compression.")

st.markdown("---")
st.write("🔹 Developed using *Python, NumPy, Matplotlib, and Streamlit*.")