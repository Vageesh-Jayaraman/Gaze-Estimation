import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="GazeFlow - Model", layout="centered")


st.title("Model Architecture")

# Display model architecture image with larger size
image = Image.open("Gaze_model.png")
st.image(image, caption="Gaze Estimation Model Architecture", use_column_width=True)

# Display clean and well-cased table of model parameters
st.markdown("### Model Parameters")
model_info = {
    "Parameter": ["Input Shape", "Optimizer", "Learning Rate", "Loss Function"],
    "Value": ["(80, 120, 3)", "Adam", "0.001", "Angular Loss"]
}
st.table(model_info)

# Neatly described Angular Loss Function
st.markdown("### Angular Loss Function")
st.markdown("""
**Angular Loss** combines directional and magnitude-based error into a single metric:

1. **Cosine Loss** – Measures the angle between predicted and true vectors. Lower is better when directions align.
2. **Mean Squared Error (MSE)** – Captures the magnitude difference between vectors.
3. **Final Formula** – The total loss is computed as: `0.7 * Cosine Loss + 0.3 * Mean Squared Error`
""")
