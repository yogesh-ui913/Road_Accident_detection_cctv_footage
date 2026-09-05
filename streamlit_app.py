import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Accident Detection from CCTV Footage",
    page_icon="🚗",
    layout="centered"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model_vgg16_aug_fine_tune.keras"
    )


model = load_model()


# --------------------------------------------------
# Image Size
# --------------------------------------------------

IMG_WIDTH = 224
IMG_HEIGHT = 224


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🚗 Accident Detection from CCTV Footage")

st.write(
    "Upload a CCTV image to predict whether it contains "
    "an Accident or Non Accident."
)


# --------------------------------------------------
# Upload Image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    # Open uploaded image
    img = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Space
    st.write("")

    # Predict button
    if st.button("🔍 Detect Accident"):

        with st.spinner("Analyzing image..."):

            try:

                # ------------------------------------------
                # Resize image
                # ------------------------------------------

                img_resized = img.resize(
                    (IMG_WIDTH, IMG_HEIGHT)
                )


                # ------------------------------------------
                # Convert image to array
                # ------------------------------------------

                img_array = image.img_to_array(
                    img_resized
                )


                # ------------------------------------------
                # Add batch dimension
                # ------------------------------------------

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )


                # ------------------------------------------
                # Normalize image
                # ------------------------------------------

                img_array = img_array / 255.0


                # ------------------------------------------
                # Prediction
                # ------------------------------------------

                prediction = model.predict(
                    img_array,
                    verbose=0
                )


                probability = float(
                    prediction[0][0]
                )


                # ------------------------------------------
                # Class Mapping
                # ------------------------------------------
                # 0 = Accident
                # 1 = Non Accident

                if probability >= 0.5:

                    predicted_class = "Non Accident"

                    confidence = probability

                else:

                    predicted_class = "Accident"

                    confidence = 1 - probability


                # ------------------------------------------
                # Display Result
                # ------------------------------------------

                st.subheader("Prediction")


                if predicted_class == "Accident":

                    st.error(
                        "🚨 Accident Detected"
                    )

                else:

                    st.success(
                        "✅ Non Accident"
                    )


                st.write(
                    f"**Prediction:** {predicted_class}"
                )

                st.info(
                    f"**Confidence:** {confidence:.2%}"
                )


            except Exception as e:

                st.error(
                    "❌ Error while processing the image."
                )

                st.exception(e)


# --------------------------------------------------
# Instructions
# --------------------------------------------------

st.markdown("---")

st.subheader("📌 Instructions")

st.write(
    "1. Upload a CCTV image."
)

st.write(
    "2. Click the 'Detect Accident' button."
)

st.write(
    "3. The model will predict Accident or Non Accident."
)
