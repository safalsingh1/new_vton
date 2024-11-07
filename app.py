import os
import streamlit as st
from gradio_client import Client, file
import tempfile
import google.generativeai as genai


# Configure the Gemini API with the API key from environment variables
genai.configure(api_key=os.environ["API_KEY"])

st.set_page_config(layout="wide")

def get_chatbot_response(user_message):
    """Fetch response from the Gemini API based on the user's message."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(user_message)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize the Gradio client with your app's endpoint
client = Client("yisol/IDM-VTON")

# Streamlit app UI
st.title("Virtual Try-On (VTON) App")
st.write("Select a sample human image and garment, or upload your own!")

def load_images_from_directory(directory):
    """Load image files from a given directory."""
    return {img: os.path.join(directory, img) for img in os.listdir(directory) if img.endswith(('.jpg', '.jpeg', '.png'))}

# Define directories for sample images
human_images_dir = "example"
garment_images_dir = "sample_garments"

local_human_images = load_images_from_directory(human_images_dir)
local_garment_images = load_images_from_directory(garment_images_dir)

col1, col2, col3 = st.columns([1, 3, 3])

# Left column: Chatbot
with col1:
    st.markdown("<div style='background-color: #f0f4ff; padding: 10px; border-radius: 5px;'><h3 style='margin: 0;'>Fashion Chatbot 🤖</h3></div>", unsafe_allow_html=True)
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = ""

    user_input = st.text_input("Ask the chatbot:", key="chat_input")

    if st.button("Send", key="send_btn"):
        response = get_chatbot_response(user_input)
        st.session_state.chat_history += f"You: {user_input}\nBot: {response}\n\n"
    st.text_area("Chat History", st.session_state.chat_history, height=300)

# Middle column: Image selection
with col2:
    st.subheader("Selected Images")

    st.write("**Human Image**")
    sample_human = st.selectbox("Select a human image:", list(local_human_images.keys()), key="human_select")
    st.image(local_human_images[sample_human], caption="Selected Human Image", use_column_width=True)

    st.write("**Garment Image**")
    sample_garment = st.selectbox("Select a garment image:", list(local_garment_images.keys()), key="garment_select")
    st.image(local_garment_images[sample_garment], caption="Selected Garment Image", use_column_width=True)

# Right column: Controls
with col3:
    st.subheader("Upload Images")
    human_image = st.file_uploader("Upload your own human image", type=["png", "jpg", "jpeg"], key="human_upload")
    garment_image = st.file_uploader("Upload your own garment image", type=["png", "jpg", "jpeg"], key="garment_upload")

    garment_desc = st.text_input("Garment Description", value="A stylish outfit")

# Sidebar for Advanced Options
with st.sidebar:
    st.subheader("Advanced Options")
    is_checked = st.checkbox("Enable Option 1 (is_checked)", value=True)
    is_checked_crop = st.checkbox("Enable Cropping (is_checked_crop)", value=False)
    denoise_steps = st.slider("Denoise Steps", min_value=0, max_value=50, value=30)
    seed = st.number_input("Seed", min_value=0, value=42)

    def save_uploaded_file(uploaded_file):
        """Save uploaded file to a temporary directory and return the file path."""
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path

    if st.button("Try On Garment", key="tryon_btn"):
        human_image_path = save_uploaded_file(human_image) if human_image else local_human_images[sample_human]
        garment_image_path = save_uploaded_file(garment_image) if garment_image else local_garment_images[sample_garment]

        if human_image_path and garment_image_path:
            with st.spinner("Generating output..."):
                try:
                    result = client.predict(
                        dict={
                            "background": file(human_image_path),
                            "layers": [],
                            "composite": None
                        },
                        garm_img=file(garment_image_path),
                        garment_des=garment_desc,
                        is_checked=is_checked,
                        is_checked_crop=is_checked_crop,
                        denoise_steps=denoise_steps,
                        seed=seed,
                        api_name="/tryon"
                    )

                    st.image(result[0], caption="Output Image", use_column_width=True)
                    st.image(result[1], caption="Masked Output Image", use_column_width=True)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.error("Please select or upload both the human and garment images.")
