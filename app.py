import os
import pickle
import numpy as np
import gradio as gr
import tensorflow as tf
from gradio.themes.base import Base
from tensorflow.keras.preprocessing.sequence import pad_sequences                                 # type: ignore

File_Path = "/content/drive/MyDrive/Shaayer/Shaayer_Data/"

# Load the Trained Model
model = tf.keras.models.load_model(f"{File_Path}RomanUrduPoetryModel.keras")

# Load Poetry Tokenizer
with open(f"{File_Path}RomanUrduPoetryTokenizer.pkl", "rb") as File:
    PoetryTokenizer = pickle.load(File)

# Set Max Sequence Length
Max_Sequence_Length = 50

def Generate_Poetry(Seed_Text, Poetry_Words_Length=20, Temperature=0.5):
    Token_List = PoetryTokenizer.texts_to_sequences([Seed_Text])[0]

    for i in range(Poetry_Words_Length):

        Token_List = pad_sequences([Token_List], maxlen=Max_Sequence_Length - 1, padding="pre")

        Predictions = model.predict(Token_List, verbose=0)[0]
        Predictions = np.log(Predictions + 1e-7) / Temperature
        Exp_Preds = np.exp(Predictions)
        Probabilities = Exp_Preds / np.sum(Exp_Preds)

        Predicted = np.random.choice(len(Probabilities), p=Probabilities)

        output_word = next((word for word, index in PoetryTokenizer.word_index.items() if index == Predicted), "")
        if not output_word:
            break

        Seed_Text += " " + output_word
        Token_List = np.append(Token_List, Predicted)                           # Update token list

    # Create Genertated Poetry File
    File_Path = "Generated_Poetry.txt"
    with open(File_Path, "w", encoding="utf-8") as File:
        File.write(Seed_Text)

    return Seed_Text, File_Path

# Customize UI
class Seafoam(Base):
    pass
seafoam = Seafoam(font=gr.themes.GoogleFont("Plus Jakarta Sans"))

style ="""
    .gradio-primary-button {
        background: #007bff;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 20px;
    }
    .gradio-primary-button:hover {
        background: #0056b3;
    }
    .gradio-dropdown {
        background: #00000000;
    }
    .gradio-secondary-button {
        background: transparent;
        border: 1.5px solid var(--input-border-color);
        font-weight: bold;
        border-radius: 20px;
    }
    .gradio-secondary-button:hover {
        background: var(--input-border-color);
    }
    label.container.show_textbox_border.svelte-173056l textarea.svelte-173056l {
        background:transparent;
        border-radius: 20px;
    }
    div.svelte-633qhp {
        border-radius: 15px;
        overflow-y: hidden;
    }
    span.svelte-1gfkn6j {
        padding-left: 20px,
        font-size:16px;
        font-weight: bold;
    }
    .gradio-container.gradio-container-5-16-0 .contain span.svelte-1gfkn6j {
        padding-left: 12px;
    }
    .icon-button-wrapper.hide-top-corner.svelte-1jx2rq3 {
        border-radius: 20px;
        margin: 5px 6.09px 0px 0px;
        padding: 6px 5.5px 5px 5.5px;
    }
    label.svelte-173056l.svelte-173056l {
        display: block;
        width: 100%;
        padding-left: 10px;
    }
"""

# Gradio Interface with Better UI
with gr.Blocks(theme=seafoam, css=style) as app:

    gr.Markdown("# Shaayer")

    with gr.Row():

        seed_input = gr.Textbox(label="Poetry Seed", placeholder="Enter your poetry seed here ...")
        num_words = gr.Slider(10, 50, step=5, label="Number of Words", value=20)
        temp = gr.Slider(0.2, 1.0, step=0.1, label="Creativity (Temperature)", value=0.5)

    poetry_output = gr.Textbox(label="Generated Poetry", placeholder="Generated Poetry will appear here ...")
    download_btn = gr.DownloadButton("Download Generated Poetry", value="generated_poetry.txt", visible=False, elem_classes=["gradio-secondary-button"])

    generate_button = gr.Button("Generate", variant="primary", elem_classes=["gradio-primary-button"])

    def generate_download_links(seed_input, num_words, temp):
        poetry_output, text_file = Generate_Poetry(seed_input, num_words, temp)
        return poetry_output, gr.update(value=text_file, visible=True)

    generate_button.click(generate_download_links, inputs=[seed_input, num_words, temp], outputs=[poetry_output, download_btn])

app.launch(share=True, inbrowser=True)