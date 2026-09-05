
import streamlit as st
import requests


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Accident Detection from CCTV Footage",
    page_icon="🚗",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🚗 Accident Detection from CCTV Footage")

st.write(
    "Upload a CCTV image to predict whether it contains "
    "an Accident or Non Accident."
)


# --------------------------------------------------
# Flask API URL
# --------------------------------------------------

FLASK_API_URL = "http://127.0.0.1:5000/predict"


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

    # Display uploaded image
    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.write("")

    # Predict button
    if st.button("🔍 Detect Accident"):

        with st.spinner("Analyzing image..."):

            try:

                # Reset file position
                uploaded_file.seek(0)

                # Send image to Flask API
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    FLASK_API_URL,
                    files=files,
                    timeout=60
                )


                # --------------------------------------------------
                # Successful response
                # --------------------------------------------------

                if response.status_code == 200:

                    prediction_data = response.json()

                    predicted_class = prediction_data.get(
                        "predicted_class",
                        "Unknown"
                    )

                    confidence = float(
                        prediction_data.get(
                            "confidence",
                            0
                        )
                    )


                    st.subheader("Prediction")


                    # Accident
                    if predicted_class.lower() == "accident":

                        st.error(
                            f"🚨 Accident Detected"
                        )

                    # Non Accident
                    elif predicted_class.lower() == "non accident":

                        st.success(
                            f"✅ Non Accident"
                        )

                    # Unknown
                    else:

                        st.warning(
                            f"Prediction: {predicted_class}"
                        )


                    # Confidence
                    st.info(
                        f"Confidence: {confidence:.2%}"
                    )


                # --------------------------------------------------
                # API Error
                # --------------------------------------------------

                else:

                    try:
                        error_message = response.json().get(
                            "error",
                            "Unknown API error"
                        )
                    except Exception:
                        error_message = response.text

                    st.error(
                        f"❌ API Error: {error_message}"
                    )


            # --------------------------------------------------
            # Flask connection error
            # --------------------------------------------------

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to Flask API."
                )

                st.info(
                    "Please start the Flask API first:"
                )

                st.code(
                    "python app.py"
                )


            # --------------------------------------------------
            # Timeout error
            # --------------------------------------------------

            except requests.exceptions.Timeout:

                st.error(
                    "⏳ Request timed out. "
                    "Please check whether the Flask API is running correctly."
                )


            # --------------------------------------------------
            # Other errors
            # --------------------------------------------------

            except Exception as e:

                st.error(
                    f"❌ Unexpected error: {e}"
                )


# --------------------------------------------------
# Instructions
# --------------------------------------------------

st.markdown("---")

st.markdown(
    
### How to Run Locally


