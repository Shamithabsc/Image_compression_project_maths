 # 📦 Image Compression using SVD with Sidebar Tabs + Formula Explanation
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float

# ============================================================
# 🔰 App Title and Introduction
# ============================================================
st.title("📸 Image Compression using SVD (Singular Value Decomposition)")

st.write(r"""
This interactive app demonstrates **Image Compression** using **SVD (Singular Value Decomposition)**  
and helps you understand how an image can be represented as a matrix and mathematically compressed.

---

### 🧠 **What is SVD?**
For any image (matrix) `A`, we can decompose it as:

\[
A = U \Sigma V^T
\]

Where:
- **U** → Matrix of left singular vectors  
- **Σ (Sigma)** → Diagonal matrix of singular values  
- **Vᵗ** → Matrix of right singular vectors  

By keeping only the **top k singular values**, we can approximate the original image with much fewer numbers.

\[
A_k = U_k \Sigma_k V_k^T
\]

This approximation reduces data storage while keeping most visual details intact.
""")

# ============================================================
# 🔹 Sidebar Navigation
# ============================================================
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to:", ["🧩 Image Compression", "📊 SVD Components", "🧮 Formula Explanation"])

# ============================================================
# 📤 Upload Image
# ============================================================
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = img_as_float(io.imread(uploaded_file))

    # Handle different image formats
    if image.ndim == 2:
        st.sidebar.info("Detected grayscale image.")
        image_rgb = np.stack((image,) * 3, axis=-1)
    elif image.shape[2] == 4:
        st.sidebar.info("Detected RGBA image (with transparency). Converting to RGB.")
        image_rgb = image[..., :3]
    else:
        image_rgb = image

    # ============================================================
    # 🧩 PAGE 1: Image Compression
    # ============================================================
    if page == "🧩 Image Compression":
        st.header("🎨 Image Compression Visualization")

        st.image(image_rgb, caption="Original Image", use_container_width=True)

        k = st.slider("Select number of singular values to keep (compression level)", 5, 200, 50)

        compressed_channels = []
        for i in range(3):  # RGB
            U, S, VT = np.linalg.svd(image_rgb[:, :, i], full_matrices=False)
            S[k:] = 0
            compressed_channel = np.dot(U, np.dot(np.diag(S), VT))
            compressed_channels.append(compressed_channel)

        compressed_image = np.stack(compressed_channels, axis=2)

        st.subheader("🖼️ Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_rgb, caption="Original", use_container_width=True)
        with col2:
            st.image(np.clip(compressed_image, 0, 1), caption=f"Compressed (k={k})", use_container_width=True)

        st.info("💡 Explore the 'Formula Explanation' tab to see how the math works behind this compression!")

    # ============================================================
    # 📊 PAGE 2: SVD Components
    # ============================================================
    elif page == "📊 SVD Components":
        st.header("📊 SVD Components and Matrices")

        channel = st.selectbox("Select Color Channel", ["Red", "Green", "Blue"])
        channel_index = {"Red": 0, "Green": 1, "Blue": 2}[channel]

        U, S, VT = np.linalg.svd(image_rgb[:, :, channel_index], full_matrices=False)

        st.subheader(f"🧮 Original {channel} Channel Matrix (Pixel Values)")
        st.dataframe(np.round(image_rgb[:, :, channel_index], 3))

        st.subheader(f"📘 U Matrix ({U.shape[0]}x{U.shape[1]})")
        st.dataframe(np.round(U, 3))

        st.subheader(f"📗 S (Singular Values) Vector ({S.shape[0]})")
        st.dataframe(np.round(S, 3))

        st.subheader(f"📙 Vᵗ Matrix ({VT.shape[0]}x{VT.shape[1]})")
        st.dataframe(np.round(VT, 3))

    # ============================================================
    # 🧮 PAGE 3: Formula Explanation with Real Values
    # ============================================================
    elif page == "🧮 Formula Explanation":
        st.header("🧮 How SVD Compression Works — Step by Step")

        st.write(r"""
We'll pick a small **3×3 section** of the image matrix (Red channel)  
and show how SVD reconstructs it using the formula:

\[
A = U \Sigma V^T
\]
        """)

        channel = image_rgb[:, :, 0]  # Red channel
        small_patch = channel[:3, :3]
        U, S, VT = np.linalg.svd(small_patch, full_matrices=False)

        st.subheader("🔹 Step 1: Original Matrix (A)")
        st.dataframe(np.round(small_patch, 3))

        st.subheader("🔹 Step 2: Decomposed Matrices (U, Σ, Vᵗ)")
        st.write("**U Matrix:**")
        st.dataframe(np.round(U, 3))

        Sigma = np.diag(S)
        st.write("**Σ (Diagonal Matrix of Singular Values):**")
        st.dataframe(np.round(Sigma, 3))

        st.write("**Vᵗ Matrix:**")
        st.dataframe(np.round(VT, 3))

        reconstructed = U @ Sigma @ VT

        st.subheader("🔹 Step 3: Reconstructed Matrix (UΣVᵗ)")
        st.dataframe(np.round(reconstructed, 3))

        st.success("✅ The reconstructed matrix is very close to the original — proving the SVD formula works!")

else:
    st.info("👆 Upload an image from the **sidebar** to begin.")
