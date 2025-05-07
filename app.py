import gradio as gr
import joblib

# Load the trained model
model = joblib.load("text_classifier.joblib")

# Class labels (adjust if your model uses different ones)
labels = ["Baseball", "Space"]

# Prediction function
def classify(text):
    prediction = model.predict([text])[0]
    return labels[prediction]

# Gradio interface
interface = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=3, placeholder="Enter your text here..."),
    outputs="text",
    title="Text Topic Classifier",
    description="This AI model predicts whether your text is about space or baseball."
)

# Run the app
if __name__ == "__main__":
    interface.launch()
