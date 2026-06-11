
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms
from torchvision.models import efficientnet_b0

st.set_page_config(
    page_title="Plant Disease Detection",
    layout="wide"
)

import json

with open("E:\\Laptop\\Final_project\\Notebooks\\class_names.json", "r") as f:
    class_names = json.load(f)


#Defining clas names
# class_names.py

# #CLASS_NAMES = [
#     'Pepper__bell___Bacterial_spot',
#     'Pepper__bell___healthy',
#     'Potato___Early_blight',
#     'Potato___healthy',
#     'Potato___Late_blight',
#     'Tomato_Bacterial_spot',
#     'Tomato_Early_blight',
#     'Tomato_healthy',
#     'Tomato_Late_blight',
#     'Tomato_Leaf_Mold',
#     'Tomato_Septoria_leaf_spot',
#     'Tomato_Spider_mites_Two_spotted_spider_mite',
#     'Tomato__Target_Spot',
#     'Tomato__Tomato_mosaic_virus',
#     'Tomato__Tomato_YellowLeaf__Curl_Virus'
# ]

recommendations = {

    "Pepper__bell___healthy":
        "Plant is healthy.",

    "Potato___Early_blight":
        "Apply Mancozeb fungicide.",

    "Potato___Late_blight":
        "Use copper-based fungicides.",

    "Tomato_Early_blight":
        "Apply fungicide and remove infected leaves.",

    "Tomato_Late_blight":
        "Use chlorothalonil-based fungicides.",

    "Tomato__Tomato_YellowLeaf__Curl_Virus":
        "Control whiteflies and remove infected plants.",

    "Tomato__Tomato_mosaic_virus":
        "Remove infected plants and sanitize tools."
}

@st.cache_resource
def load_model():

    model = efficientnet_b0()

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        15
    )

    model.load_state_dict(
        torch.load(
            "E:\\Laptop\\Final_project\\Notebooks\\best_efficientnet.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def predict(image):

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, pred = torch.max(
            probabilities,
            dim=1
        )

    return (
        class_names[pred.item()],
        confidence.item() * 100
    )

st.title(
    "🌿 Plant Disease Detection"
)

st.write(
    "Upload a plant leaf image to detect disease."
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)


if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image"
    )

    if st.button("Predict Disease"):

        disease, confidence = predict(image)

        st.success(
            f"Detected: {disease}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        recommendation = recommendations.get(
            disease,
            "Consult agricultural expert."
        )

        st.warning(
            f"Suggested Action: {recommendation}"
        )