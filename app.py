import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
# from tensorflow.keras.applications.efficientnet import preprocess_input

import inspect
print(inspect.getsource(tf.keras.applications.efficientnet.preprocess_input))


# @st.cache_resource
# def load_model():
#     model = tf.keras.models.load_model("best_model.h5")
#     return model

# model = load_model()

print(tf.keras.__version__)
print(tf.__version__)

model_path = hf_hub_download(
    repo_id="RawanAlwadeya/TransferLearningDR",
    filename="TransferLearningDR.h5"
)
model = tf.keras.models.load_model(model_path)


def preprocess_image(image):
    IMG_SIZE = (224, 224)
    image = image.convert("RGB")  
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)      
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Detection"])


if page == "Home":
    st.markdown("<h1 style='text-align: center;'>🩺 Diabetic Retinopathy Detection App</h1>",
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Transfer Learning–Powered Retinal Image Classification</h3>",
                unsafe_allow_html=True)

    st.write(
        """
        **Diabetic Retinopathy (DR)** is a diabetes-related eye condition that damages the
        blood vessels in the retina, potentially leading to vision loss if untreated.  
        
        Early detection through retinal imaging is critical to prevent complications.

        This app leverages **Transfer Learning** to classify retinal images as either:
        - **⚠️ Diabetic Retinopathy Detected**  
        - **✅ No Diabetic Retinopathy**
        """
    )

    st.image("Diabetic_Retinopathy.jpg",
        caption="Retinal Image Example: Healthy vs DR",
        use_container_width=True)

    st.info("👉 Go to the **Detection** page from the left sidebar to upload a retinal image and get predictions.")


elif page == "Detection":
    st.markdown("<h1 style='text-align: center;'>🩺 Diabetic Retinopathy Detection</h1>",
                unsafe_allow_html=True)
    st.write(
        "Upload a retinal fundus image below, and the model will predict whether "
        "it shows **Diabetic Retinopathy (DR)** or **No DR**."
    )

    uploaded_file = st.file_uploader("Upload Retinal Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Retinal Image", use_container_width=True)

        img_array = preprocess_image(image)
        prediction = model.predict(img_array)[0][0]

            
        if prediction < 0.5:     
            st.error("⚠️ The model predicts **Diabetic Retinopathy likely detected**. "
                "Please consult an eye-care professional for confirmation.")
        else:
            st.success("✅ The model predicts **Likely No Diabetic Retinopathy**. "
                "For medical certainty, always consult a professional.")


