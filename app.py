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

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(
        135deg,
        #667eea,
        #764ba2,
        #89f7fe,
        #66a6ff
    );
    background-size: 300% 300%;
    animation: bgFlow 18s ease infinite;
    overflow: hidden;
    font-family: 'Segoe UI', sans-serif;
}

/* BACKGROUND ANIMATION */
@keyframes bgFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* FLOATING SHAPES */
.shape {
    position: fixed;
    bottom: -20px;
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
        transform: translateY(-120vh);
        opacity: 0.8;
    }
}

/* SMALL COLORED BALLS */
.ball {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    opacity: 0.7;
}

.ball.blue { background: #81d4fa; }
.ball.pink { background: #f48fb1; }
.ball.purple { background: #ce93d8; }
.ball.green { background: #a5d6a7; }
.ball.yellow { background: #fff59d; }
.ball.orange { background: #ffcc80; }

/* BALL POSITIONS */
.ball:nth-child(1) { left: 5%; animation-duration: 14s; }
.ball:nth-child(2) { left: 15%; animation-duration: 18s; }
.ball:nth-child(3) { left: 30%; animation-duration: 12s; }
.ball:nth-child(4) { left: 45%; animation-duration: 20s; }
.ball:nth-child(5) { left: 60%; animation-duration: 16s; }
.ball:nth-child(6) { left: 75%; animation-duration: 22s; }
.ball:nth-child(7) { left: 90%; animation-duration: 15s; }

/* TITLE */
.title {
    font-size: 42px;
    font-weight: 900;
    text-align: center;
    color: #ffffff;
    text-shadow: 0 0 12px rgba(0,0,0,0.4);
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #f1f1f1;
    margin-bottom: 30px;
}

/* UPLOADER */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.9);
    border-radius: 20px;
    border: 2px dashed #ffffff;
    padding: 20px;
}

/* BUTTON */
.stButton button {
    background: linear-gradient(135deg, #ff9a9e, #fad0c4);
    color: #4a148c;
    font-size: 18px;
    font-weight: bold;
    border-radius: 40px;
    padding: 14px 40px;
    border: none;
    box-shadow: 0 0 18px rgba(0,0,0,0.3);
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.08);
}

/* IMAGE */
img {
    border-radius: 20px;
    box-shadow: 0 0 25px rgba(0,0,0,0.3);
}

/* CAPTION BOX */
.caption-box {
    margin-top: 25px;
    padding: 22px;
    border-radius: 25px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    background: rgba(255,255,255,0.95);
    color: #4a148c;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #ffffff;
    margin-top: 40px;
}

</style>

<!-- FLOATING BALLS -->
<div class="shape ball blue"></div>
<div class="shape ball pink"></div>
<div class="shape ball purple"></div>
<div class="shape ball green"></div>
<div class="shape ball yellow"></div>
<div class="shape ball orange"></div>
<div class="shape ball blue"></div>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown('<div class="title">🖼️ AI Image Caption Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image and let AI describe it ✨</div>', unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model

processor, model = load_model()

# ------------------ IMAGE UPLOAD ------------------
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("✨ Generate Caption"):
        with st.spinner("AI is thinking... 🤖"):
            inputs = processor(image, return_tensors="pt")
            output = model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)

        st.markdown(
            f'<div class="caption-box">📸 {caption}</div>',
            unsafe_allow_html=True
        )

# ------------------ FOOTER ------------------
st.markdown('<div class="footer">Made with  using Streamlit & BLIP</div>', unsafe_allow_html=True)



