
""" Image generation component"""

from diffusers import StableDiffusionPipeline
import torch
import os
import asyncio


# Load the SD Turbo model from Hugging Face
model_id = "stabilityai/sd-turbo"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float32
).to("cpu")

# Improve CPU performance
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

async def async_generate_image(headline: str):
    return await asyncio.to_thread(generate_image_from_headline, headline)

def generate_image_from_headline(headline: str, out_dir: str = "generated_images"):
    os.makedirs(out_dir, exist_ok=True)
    prompt = f"Photo realistic illustration: {headline}"
    
    image = pipe(prompt, num_inference_steps=4, guidance_scale=1.5).images[0]

    # Sanitize filename
    filename = os.path.join(
        out_dir,
        headline.lower().replace(" ", "_").replace(":", "").replace("?", "")[:100] + ".png"
    )
    
    image.save(filename)
    print(f"[✓] Saved: {filename}")
    return filename

# 🧪 Example usage
if __name__ == "__main__":
    headline = "Trump says Israel has agreed to conditions for 60-day Gaza ceasefire"
    generate_image_from_headline(headline)