import gradio as gr
from local_model import query_local
from remote_model import query_remote, pipe
import time

def query(image, question, model_name):
    if model_name == "Local":
        return query_local(image, question)
    elif model_name == "Remote":
        return query_remote(image, question, pipe)
    return "No model selected"


custom_css = """
.output-card {
    background-color: #f9fafb; 
    border: 10px solid #e5e7eb; 
    border-radius: 8px; 
    padding: 40px; 
}
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Qwen2-VL Analyst") as app:
    
    start_time = time.time()
    
    gr.Markdown(
        r"""
        ¯\_(ツ)_/¯ Intelligence: Upload an image and ask a question
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload Image", height=400)
            q_input = gr.Textbox(label="Question", lines=2)
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Analyze Image", variant="primary")
        with gr.Column(scale=1):
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="Select Model", choices=["Local", "Remote"], value="Local"
                )
            gr.Markdown("Model Analysis:")

            with gr.Group(elem_classes="output-card"):
                output_box = gr.Markdown(value="Results...", line_breaks=True)

    submit_btn.click(
        fn=query, inputs=[img_input, q_input, model_dropdown], outputs=output_box
    )



    q_input.submit(
        fn=query, inputs=[img_input, q_input, model_dropdown], outputs=output_box
    )



    def clear_inputs():
        return None, "", ""

    clear_btn.click(
        fn=clear_inputs, inputs=[], outputs=[img_input, q_input, output_box]
    )


app.launch()
