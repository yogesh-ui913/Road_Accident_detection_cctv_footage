import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Accident Detection from CCTV Footage",
    page_icon="🚗",
    layout="centered"
)


# ==================================================
# TITLE
# ==================================================

st.title("🚗 Accident Detection from CCTV Footage")

st.write(
    "Upload a CCTV image to predict whether it contains "
    "an Accident or Non Accident."
)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():

    try:

        model = tf.keras.models.load_model(
            "model_vgg16_aug_fine_tune.keras",
            compile=False
        )

        return model

    except Exception as e:

        st.error("❌ Model could not be loaded.")

        st.code(str(e))

        st.info(
            "Please check that model_vgg16_aug_fine_tune.keras "
            "exists in the GitHub repository."
        )

        st.stop()


model = load_model()


# ==================================================
# IMAGE SIZE
# ==================================================

IMG_WIDTH = 224
IMG_HEIGHT = 224


# ==================================================
# UPLOAD IMAGE
# ==================================================

uploaded_file = st.file_uploader(
    "Choose a CCTV image...",
    type=["jpg", "jpeg", "png"]
)


# ==================================================
# PREDICTION
# ==================================================

if uploaded_file is not None:

    try:

        # ------------------------------------------
        # Open image
        # ------------------------------------------

        img = Image.open(uploaded_file).convert("RGB")


        # ------------------------------------------
        # Display image
        # ------------------------------------------

        st.image(
            img,
            caption="Uploaded Image",
            use_container_width=True
        )


        st.write("")


        # ------------------------------------------
        # Detect button
        # ------------------------------------------

        if st.button("🔍 Detect Accident"):

            with st.spinner("Analyzing image..."):

                # ----------------------------------
                # Resize
                # ----------------------------------

                img_resized = img.resize(
                    (IMG_WIDTH, IMG_HEIGHT)
                )


                # ----------------------------------
                # Convert to NumPy array
                # ----------------------------------

                img_array = np.array(
                    img_resized
                ).astype("float32")


                # ----------------------------------
                # Normalize
                # Same as training
                # ----------------------------------

                img_array = img_array / 255.0


                # ----------------------------------
                # Add batch dimension
                # ----------------------------------

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )


                # ----------------------------------
                # Prediction
                # ----------------------------------

                prediction = model.predict(
                    img_array,
                    verbose=0
                )


                probability = float(
                    prediction[0][0]
                )


                # ==================================
                # CLASS MAPPING
                # ==================================
                #
                # 0 = Accident
                # 1 = Non Accident
                #
                # probability >= 0.5
                #       -> Non Accident
                #
                # probability < 0.5
                #       -> Accident
                # ==================================

                if probability >= 0.5:

                    predicted_class = "Non Accident"

                    confidence = probability

                else:

                    predicted_class = "Accident"

                    confidence = 1 - probability


                # ----------------------------------
                # Display prediction
                # ----------------------------------

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
                    f"**Confidence:** "
                    f"{confidence:.2%}"
                )


                st.write(
                    f"**Model Output:** "
                    f"{probability:.4f}"
                )


    except Exception as e:

        st.error(
            "❌ Error while processing the image."
        )

        st.code(str(e))


# ==================================================
# INSTRUCTIONS
# ==================================================

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
