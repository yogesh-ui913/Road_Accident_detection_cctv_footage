
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Accident Detection",
    page_icon="🚗"
)


# --------------------------------------------------
# Load trained VGG16 model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model_vgg16_aug_fine_tune.keras"
    )


model = load_model()


# --------------------------------------------------
# Image size used during training
# --------------------------------------------------

IMG_HEIGHT = 224
IMG_WIDTH = 224


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🚗 Accident Detection from CCTV Image")

st.write(
    "Upload a CCTV image to detect whether it contains "
    "an Accident or Non Accident."
)


# --------------------------------------------------
# Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    # Open image
    img = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    # --------------------------------------------------
    # Preprocessing
    # Same as notebook
    # --------------------------------------------------

    img = img.resize((IMG_WIDTH, IMG_HEIGHT))

    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Training used rescale=1./255
    img_array = img_array / 255.0

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    if st.button("🔍 Detect Accident"):

        with st.spinner("Analyzing image..."):

            prediction = model.predict(
                img_array,
                verbose=0
            )

        probability = float(prediction[0][0])

        # --------------------------------------------------
        # IMPORTANT:
        #
        # 0 = Accident
        # 1 = Non Accident
        #
        # probability >= 0.5 → Non Accident
        # probability < 0.5  → Accident
        # --------------------------------------------------

        if probability >= 0.5:

            predicted_class = "Non Accident"
            confidence = probability

        else:

            predicted_class = "Accident"
            confidence = 1 - probability

        # --------------------------------------------------
        # Display result
        # --------------------------------------------------

        st.subheader("Prediction")

        if predicted_class == "Accident":

            st.error("🚨 Accident Detected")

        else:

            st.success("✅ Non Accident")

        st.write(
            f"**Prediction:** {predicted_class}"
        )

        st.write(
            f"**Confidence:** {confidence:.2%}"
        )

        st.write(
            f"**Raw model output:** {probability:.4f}"
        )
