import torch
import gradio as gr
from PIL import Image
from consts import BASE_MODEL
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


"""
Initalize Model
"""
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Qwen2VLForConditionalGeneration.from_pretrained(BASE_MODEL)
processor = AutoProcessor.from_pretrained(BASE_MODEL)


"""
Model Function
"""
def query(image: Image.Image, question: str):
    if image is None:
        return "Upload an image bro."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    images, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=text, 
        images=images, 
        videos=video_inputs,
        padding=True, 
        return_tensors="pt")

    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=256)

    # Trim the input tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # Decode the output
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )

    return output_text[0]



"""
Interface
"""

custom_css = """
.output-card {
    background-color: #f9fafb; 
    border: 10px solid #e5e7eb; 
    border-radius: 8px; 
    padding: 40px; 
}
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Qwen2-VL Analyst") as app:
    
    # Header
    gr.Markdown(
        """
        ¯\(ツ)/¯ Intelligence: Upload an image and ask a question
        """
    )
    
    with gr.Row():
        # Inputs
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload Image", height=400)
            q_input = gr.Textbox(
                label="Question", 
                lines=2
            )
            
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Analyze Image", variant="primary")

        # Output
        with gr.Column(scale=1):
            gr.Markdown("Model Analysis:")
            
            with gr.Group(elem_classes="output-card"):
                output_box = gr.Markdown(
                    value="Results...", 
                    line_breaks=True
                )

    # Trigger on Button Click
    submit_btn.click(
        fn=query, 
        inputs=[img_input, q_input], 
        outputs=output_box
    )
    
    # Trigger on pressing Enter
    q_input.submit(
        fn=query, 
        inputs=[img_input, q_input], 
        outputs=output_box
    )

    # Clear button
    def clear_inputs():
        return None, "", ""
        
    clear_btn.click(fn=clear_inputs, inputs=[], outputs=[img_input, q_input, output_box])


app.launch()