import gradio as gr
import joblib

# Load the trained model
model = joblib.load("text_classifier.joblib")
labels = ["Baseball", "Space Science"]

def classify_text(text):
    prediction = model.predict([text])[0]
    return labels[prediction]

interface = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=3, placeholder="Enter text here..."),
    outputs="text",
    title="Text Topic Classifier",
    description="Classifies text as related to Baseball or Space Science."
)

interface.launch(share=True)
