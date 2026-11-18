import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "tomato_cnn_final.keras"
INDICES_PATH = "class_indices.json"
IMG_SIZE = (224, 224)          # fallback; the app reads the real size from the model
CONFIDENCE_THRESHOLD = 60.0

DISEASES = {
    "bacterial": ("Bacterial Spot", "Remove infected leaves, avoid overhead watering, and use certified disease-free seed."),
    "early": ("Early Blight", "Remove lower infected leaves, rotate crops, and keep leaves dry. Consult an agriculture officer for a suitable fungicide."),
    "healthy": ("Healthy", "The leaf looks healthy. Keep monitoring regularly and maintain good field hygiene."),
    "late": ("Late Blight", "Spreads fast in cool, wet weather. Remove infected plants quickly and contact your local agriculture officer."),
    "mold": ("Leaf Mold", "Improve air circulation, reduce humidity and avoid wetting the leaves."),
    "mosaic": ("Mosaic Virus", "No cure. Remove and destroy infected plants, disinfect tools and wash hands before handling healthy plants."),
    "septoria": ("Septoria Leaf Spot", "Remove infected lower leaves, mulch the soil, avoid overhead watering and rotate crops."),
    "spider": ("Spider Mites", "Spray water on the underside of leaves, avoid dusty and dry conditions, and consult an expert for miticide use."),
    "target": ("Target Spot", "Remove infected leaves, improve air flow and avoid leaf wetness."),
    "yellow": ("Yellow Leaf Curl", "Spread by whiteflies. Control whiteflies, use yellow sticky traps and remove infected plants."),
}


def clean(raw):
    low = raw.lower()
    for key, (name, advice) in DISEASES.items():
        if key in low:
            return name, advice
    return raw.replace("_", " ").strip(), "Consult your local agriculture officer."


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_classes():
    with open(INDICES_PATH) as f:
        data = json.load(f)
    first_key = next(iter(data))
    if isinstance(data[first_key], int):
        ordered = sorted(data.items(), key=lambda kv: kv[1])
        raw_names = [k for k, _ in ordered]
    else:
        ordered = sorted(data.items(), key=lambda kv: int(kv[0]))
        raw_names = [v for _, v in ordered]
    return [clean(n) for n in raw_names]


def predict(model, image):
    shape = model.input_shape[1:3]
    size = tuple(shape) if None not in shape else IMG_SIZE
    img = image.convert("RGB").resize(size)
    arr = np.expand_dims(np.array(img, dtype="float32"), axis=0)
    return model.predict(arr, verbose=0)[0] * 100


st.set_page_config(page_title="Tomato Leaf Disease Detector", page_icon="🍅", layout="centered")
st.title("🍅 AI-Driven Tomato Leaf Disease Identification")
st.caption("Odisha University of Agriculture and Technology")

model = load_model()
classes = load_classes()

tab1, tab2 = st.tabs(["📁 Upload photo", "📷 Use camera"])
with tab1:
    uploaded = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])
with tab2:
    camera = st.camera_input("Take a photo of the leaf")

file = uploaded or camera

if file is not None:
    image = Image.open(file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Input leaf", width="stretch")

    with st.spinner("Analysing leaf..."):
        probs = predict(model, image)

    top3 = np.argsort(probs)[::-1][:3]
    label, advice = classes[top3[0]]
    conf = probs[top3[0]]

    with col2:
        if label == "Healthy":
            st.success(f"✅ {label}")
        else:
            st.error(f"⚠️ {label}")
        st.metric("Confidence", f"{conf:.2f}%")
        if conf < CONFIDENCE_THRESHOLD:
            st.warning("Low confidence. Try a clearer, close-up photo of a single leaf.")

    st.subheader("Advice")
    st.info(advice)

    st.subheader("Top 3 predictions")
    df = pd.DataFrame({
        "Class": [classes[i][0] for i in top3],
        "Confidence (%)": [round(float(probs[i]), 2) for i in top3],
    })
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={"Confidence (%)": st.column_config.NumberColumn(format="%.2f")},
    )
    st.bar_chart(df.set_index("Class"))
else:
    st.write("Upload or capture a tomato leaf photo to get a diagnosis.")

st.divider()
st.caption("This tool supports early detection. For serious outbreaks, consult your local agriculture officer.")
