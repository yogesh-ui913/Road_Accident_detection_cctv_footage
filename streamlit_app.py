import streamlit as st
import requests


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Road Accident Detection",
    page_icon="🚗",
    layout="centered"
)


# ---------------------------------------------------------
# Flask API URL
# ---------------------------------------------------------

FLASK_API_URL = "http://127.0.0.1:5000/predict"


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("🚗 Road Accident Detection")
st.write("Upload an image to detect whether it shows an accident.")


# ---------------------------------------------------------
# Upload image
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# ---------------------------------------------------------
# Display image
# ---------------------------------------------------------

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    # -----------------------------------------------------
    # Prediction button
    # -----------------------------------------------------

    if st.button("🔍 Detect Accident"):

        try:
            # Get image bytes
            image_bytes = uploaded_file.getvalue()

            # Send image to Flask API
            files = {
                "file": (
                    uploaded_file.name,
                    image_bytes,
                    uploaded_file.type
                )
            }

            with st.spinner("Analyzing image..."):

                response = requests.post(
                    FLASK_API_URL,
                    files=files,
                    timeout=60
                )

            # -------------------------------------------------
            # Successful response
            # -------------------------------------------------

            if response.status_code == 200:

                result = response.json()

                predicted_class = result.get(
                    "predicted_class",
                    "Unknown"
                )

                confidence = result.get(
                    "confidence",
                    0
                )

                # Convert confidence to percentage
                confidence_percentage = float(confidence) * 100

                # -------------------------------------------------
                # Display result
                # -------------------------------------------------

                if predicted_class.lower() == "accident":

                    st.error("🚨 Accident Detected")

                elif predicted_class.lower() in [
                    "non accident",
                    "non-accident",
                    "non_accident"
                ]:

                    st.success("✅ No Accident Detected")

                else:

                    st.warning(
                        f"Prediction: {predicted_class}"
                    )

                st.write(
                    f"**Prediction:** {predicted_class}"
                )

                st.write(
                    f"**Confidence:** "
                    f"{confidence_percentage:.2f}%"
                )

            else:

                st.error(
                    f"Flask API Error: "
                    f"{response.status_code}"
                )

                try:
                    st.json(response.json())
                except Exception:
                    st.write(response.text)

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Could not connect to Flask API."
            )

            st.info(
                "Please start Flask first using:"
            )

            st.code(
                "python app.py",
                language="bash"
            )

        except requests.exceptions.Timeout:

            st.error(
                "⏱️ Flask API took too long to respond."
            )

        except Exception as e:

            st.error(
                f"❌ Unexpected error: {str(e)}"
            )


# ---------------------------------------------------------
# Instructions
# ---------------------------------------------------------

st.markdown("---")

st.subheader("How to run the application")

st.write("**Step 1 — Start Flask API:**")

st.code(
    "python app.py",
    language="bash"
)

st.write("**Step 2 — Start Streamlit:**")

st.code(
    "streamlit run streamlit_app.py",
    language="bash"
)

st.info(
    "Make sure the Flask API is running before clicking "
    "'Detect Accident'."
)
