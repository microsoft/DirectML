import torch
import torch_directml
import gradio as gr
from diffusers import AutoPipelineForText2Image,  StableDiffusionPipeline, LMSDiscreteScheduler
from PIL import Image
import numpy as np
 
def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.
 
lms = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear"
)
 
device = torch_directml.device(torch_directml.default_device())
 
block = gr.Blocks(css=".container { max-width: 800px; margin: auto; }")
num_samples = 2
 
def load_model(model_name):
    return AutoPipelineForText2Image.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        variant="fp16"
        ).to(device)
 
model_name = "stabilityai/sd-turbo"
pipe = load_model("stabilityai/sd-turbo")
 
def infer(prompt, inference_step, model_selector):
    global model_name, pipe
 
    if model_selector == "SD Turbo":
        if model_name != "stabilityai/sd-turbo":
            model_name = "stabilityai/sd-turbo"
            pipe = load_model("stabilityai/sd-turbo")
    else:
        if model_name != "stabilityai/sdxl-turbo":
            model_name = "stabilityai/sdxl-turbo"
            pipe = load_model("stabilityai/sdxl-turbo")
        
    images = pipe(prompt=[prompt] * num_samples, num_inference_steps=inference_step, guidance_scale=0.0)[0]
    return images
 
 
with block as demo:
    gr.Markdown("<h1><center>Stable Diffusion Turbo and XL Turbo with DirectML Backend</center></h1>")
 
    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):
 
                text = gr.Textbox(
                    label="Enter your prompt", show_label=False, max_lines=1
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Run").style(
                    margin=False,
                    rounded=(False, True, True, False),
                )
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                iteration_slider = gr.Slider(
                    label="Steps",
                    step = 1,
                    maximum = 4,
                    minimum = 1,
                    value = 1         
                )
 
                model_selector = gr.Dropdown(
                    ["SD Turbo", "SD Turbo XL"], label="Model", info="Select the SD model to use", value="SD Turbo"
                )
 
        gallery = gr.Gallery(label="Generated images", show_label=False).style(
            grid=[2], height="auto"
        )
        text.submit(infer, inputs=[text, iteration_slider, model_selector], outputs=gallery)
        btn.click(infer, inputs=[text, iteration_slider, model_selector], outputs=gallery)
 
    gr.Markdown(
        """___
   <p style='text-align: center'>
   Created by CompVis and Stability AI
   <br/>
   </p>"""
    )
 
demo.launch(debug=True)