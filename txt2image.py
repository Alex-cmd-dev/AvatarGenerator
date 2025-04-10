import gradio as gr
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from stable_diffusion import StableDiffusionXL


def genereate_images_from_prompt(
    prompt,
    n_images=4,
    steps=None,
    cfg=None,
    negative_prompt="",
    n_rows=1,
    decoding_batch_size=1,
    quantize=False,
    float16=True,
    preload_models=False,
    seed=None,
):
    # Load the models
    sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=float16)
    if quantize:
        nn.quantize(
            sd.text_encoder_1, class_predicate=lambda _, m: isinstance(m, nn.Linear)
        )
        nn.quantize(
            sd.text_encoder_2, class_predicate=lambda _, m: isinstance(m, nn.Linear)
        )
        nn.quantize(sd.unet, group_size=32, bits=8)
    cfg = cfg or 0.0
    steps = steps or 2

    # Ensure that models are read in memory if needed
    if preload_models:
        sd.ensure_models_are_loaded()


    #edit the prompt
    prompt = "Generate avatar/profile picture of " + prompt
    # Generate the latent vectors using diffusion
    latents = sd.generate_latents(
        prompt,
        n_images=n_images,
        cfg_weight=cfg,
        num_steps=steps,
        seed=seed,
        negative_text=negative_prompt,
    )
    for x_t in tqdm(latents, total=steps):
        mx.eval(x_t)

    # The following is not necessary but it may help in memory
    # constrained systems by reusing the memory kept by the unet and the text
    # encoders.
    del sd.text_encoder_1
    del sd.text_encoder_2
    del sd.unet
    del sd.sampler
    peak_mem_unet = mx.get_peak_memory() / 1024**3

    # Decode them into images
    decoded = []
    for i in tqdm(range(0, n_images, decoding_batch_size)):
        decoded.append(sd.decode(x_t[i : i + decoding_batch_size]))
        mx.eval(decoded[-1])
    peak_mem_overall = mx.get_peak_memory() / 1024**3

    # Arrange them on a grid
    x = mx.concatenate(decoded, axis=0)
    x = mx.pad(x, [(0, 0), (8, 8), (8, 8), (0, 0)])
    B, H, W, C = x.shape
    x = x.reshape(n_rows, B // n_rows, H, W, C).transpose(0, 2, 1, 3, 4)
    x = x.reshape(n_rows * H, B // n_rows * W, C)
    x = (x * 255).astype(mx.uint8)

    # Return the image
    image = Image.fromarray(np.array(x))
    
    # Report the peak memory used during generation
    memory_info = f"Peak memory used for the unet: {peak_mem_unet:.3f}GB\nPeak memory used overall: {peak_mem_overall:.3f}GB"
    
    return image, memory_info


# Create Gradio UI
def create_ui():
    with gr.Blocks(title="Stable Diffusion XL with MLX") as app:
        gr.Markdown("# Stable Diffusion XL Text-to-Image Generator with MLX")
        gr.Markdown("Generate Avatars from text prompts using Stable Diffusion XL running on MLX.")
        
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="Enter negative prompt here...",
                    lines=2
                )
                
                with gr.Row():
                    n_images = gr.Slider(
                        minimum=1, 
                        maximum=16, 
                        step=1, 
                        value=4, 
                        label="Number of Images"
                    )
                    n_rows = gr.Slider(
                        minimum=1, 
                        maximum=4, 
                        step=1, 
                        value=1, 
                        label="Number of Rows"
                    )
                
                with gr.Row():
                    steps = gr.Slider(
                        minimum=1, 
                        maximum=50, 
                        step=1, 
                        value=2, 
                        label="Steps"
                    )
                    cfg = gr.Slider(
                        minimum=0, 
                        maximum=15, 
                        step=0.5, 
                        value=0, 
                        label="CFG Scale"
                    )
                
                with gr.Row():
                    decoding_batch_size = gr.Slider(
                        minimum=1, 
                        maximum=8, 
                        step=1, 
                        value=1, 
                        label="Decoding Batch Size"
                    )
                    seed = gr.Number(
                        label="Seed (blank for random)", 
                        precision=0
                    )
                
                with gr.Row():
                    quantize = gr.Checkbox(
                        label="Enable Quantization", 
                        value=False
                    )
                    float16 = gr.Checkbox(
                        label="Use Float16", 
                        value=True
                    )
                    preload_models = gr.Checkbox(
                        label="Preload Models", 
                        value=False
                    )
                
                generate_btn = gr.Button("Generate Images", variant="primary")
            
            with gr.Column(scale=3):
                output_image = gr.Image(label="Generated Image")
                memory_info = gr.Textbox(label="Memory Usage", lines=2)
        
        # Connect the function
        generate_btn.click(
            fn=genereate_images_from_prompt,
            inputs=[
                prompt,
                n_images,
                steps,
                cfg,
                negative_prompt,
                n_rows,
                decoding_batch_size,
                quantize,
                float16,
                preload_models,
                seed,
            ],
            outputs=[output_image, memory_info]
        )
        
    return app


# Launch the UI
if __name__ == "__main__":
    ui = create_ui()
    ui.launch()