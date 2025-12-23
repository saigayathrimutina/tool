import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

# ------------------ CUSTOM CSS (PASTEL SHAPES THEME) ------------------
st.markdown("""
<style>

/* MAIN BACKGROUND - LIGHT PASTEL GRADIENT */
.stApp {
    background: linear-gradient(
        135deg,
        #fde2e4,
        #e0f7fa,
        #f3e5f5,
        #fffde7
    );
    background-size: 400% 400%;
    animation: bgFlow 15s ease infinite;
    overflow: hidden;
    font-family: 'Segoe UI', sans-serif;
}

/* BACKGROUND ANIMATION */
@keyframes bgFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* FLOATING SHAPES BASE */
.shape {
    position: fixed;
    opacity: 0.5;
    animation: floatUp linear infinite;
    z-index: 0;
}

/* FLOAT ANIMATION */
@keyframes floatUp {
    from {
        transform: translateY(100vh);
        opacity: 0;
    }
    to {
        transform: translateY(-15vh);
        opacity: 0.6;
    }
}

/* DIAMOND */
.diamond {
    width: 18px;
    height: 18px;
    background: #ce93d8;
    transform: rotate(45deg);
}

/* BALL */
.ball {
    width: 18px;
    height: 18px;
    background: #81d4fa;
    border-radius: 50%;
}

/* CLOUD */
.cloud {
    width: 30px;
    height: 18px;
    background: #ffffff;
    border-radius: 20px;
    box-shadow: 
        10px 0 0 #ffffff,
        -10px 0 0 #ffffff;
}

/* DROP */
.drop {
    width: 14px;
    height: 20px;
    background: #80deea;
    border-radius: 50% 50% 50% 50%;
}

/* SHAPE POSITIONS */
.shape:nth-child(1) { left: 10%; animation-duration: 10s; }
.shape:nth-child(2) { left: 25%; animation-duration: 14s; }
.shape:nth-child(3) { left: 40%; animation-duration: 12s; }
.shape:nth-child(4) { left: 60%; animation-duration: 16s; }
.shape:nth-child(5) { left: 75%; animation-duration: 11s; }
.shape:nth-child(6) { left: 90%; animation-duration: 13s; }

/* TITLE */
.title {
    font-size: 46px;
    font-weight: 900;
    text-align: center;
    color: #6a1b9a;
    text-shadow: 0 0 10px rgba(186,104,200,0.6);
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #4a148c;
    margin-bottom: 30px;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.8);
    border-radius: 20px;
    border: 2px dashed #ba68c8;
    padding: 20px;
}

/* BUTTON */
.stButton button {
    background: linear-gradient(135deg, #ce93d8, #81d4fa);
    color: #4a148c;
    font-size: 18px;
    font-weight: bold;
    border-radius: 40px;
    padding: 14px 40px;
    border: none;
    box-shadow: 0 0 18px rgba(186,104,200,0.6);
    transition: all 0.3s ease;
}

.stButton button:hover {
    transform: scale(1.08);
    box-shadow: 0 0 30px rgba(129,212,250,0.9);
}

/* IMAGE */
img {
    border-radius: 20px;
    box-shadow: 0 0 25px rgba(0,0,0,0.25);
}

/* CAPTION BOX */
.caption-box {
    margin-top: 25px;
    padding: 25px;
    border-radius: 25px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    background: rgba(255,255,255,0.9);
    color: #4a148c;
    border: 2px solid #ce93d8;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #6a1b9a;
    margin-top: 40px;
}

</style>

<!-- FLOATING SHAPES -->
<div class="shape diamond"></div>
<div class="shape ball"></div>
<div class="shape cloud"></div>
<div class="shape drop"></div>
<div class="shape diamond"></div>
<div class="shape ball"></div>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown('<div class="title">🖼️ AI Image Caption Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image and let AI describe it beautifully ✨</div>', unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# ------------------ IMAGE UPLOAD ------------------
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="✨ Uploaded Image ✨", use_container_width=True)

    if st.button("✨ Generate Caption"):
        with st.spinner("AI is creating magic... 🤖✨"):
            inputs = processor(image, return_tensors="pt")
            output = model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)

        st.markdown(
            f'<div class="caption-box">📸 {caption}</div>',
            unsafe_allow_html=True
        )

# ------------------ FOOTER ------------------
st.markdown('<div class="footer">Made with ❤️ using Streamlit & AI</div>', unsafe_allow_html=True)
