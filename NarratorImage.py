
""" Basic narrator with Q&A and image generation"""

import asyncio
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import aioconsole
import sys



load_dotenv(override=True)

from diffusers import StableDiffusionPipeline
import torch

client = AsyncOpenAI()

# === Stable Diffusion ===
# Load the SD Turbo model from Hugging Face
model_id = "stabilityai/sd-turbo"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float32
).to("cpu")

# Improve CPU performance
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.set_progress_bar_config(disable=True)  # Hide progress bar

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
    return filename

async def async_generate_image(headline: str):
    return await asyncio.to_thread(generate_image_from_headline, headline)

headlines = [
    "Trump says Israel has agreed to conditions for 60-day Gaza ceasefire",
    "NASA plans new moon mission by 2026",
    "Global markets rally as inflation slows",
    "AI surpasses human accuracy in cancer detection",
    "Massive solar storm expected to hit Earth this week"
]

async def answer_question(question: str, context: str = "") -> str:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI that explains news headlines."},
            {"role": "user", "content": f"{context}\n\n{question}"}
        ]
    )
    return response.choices[0].message.content.strip()

# === Real-time narrator ===
current_index = 0
narration_event = asyncio.Event()
narration_event.set()
done = asyncio.Event()
print_lock = asyncio.Lock()

def safe_print(message):
    """Print with immediate flush to ensure visibility"""
    print(message, flush=True)
    sys.stdout.flush()  # Force flush

async def safe_async_print(message):
    """Async print with lock and immediate flush"""
    async with print_lock:
        safe_print(message)

async def narrator():
    global current_index
    while current_index < len(headlines) and not done.is_set():
        # Wait for narration to be enabled
        await narration_event.wait()
        
        # Check if we should exit
        if done.is_set():
            break
            
        # Double-check narration is still enabled after wait
        if not narration_event.is_set():
            continue
            
        headline = headlines[current_index]
        await safe_async_print(f"\n📰 Headline {current_index + 1}: {headline}")
        await safe_async_print("🛠️ Generating image...")

        # Generate image (this might take a while)
        try:
            image_path = await asyncio.to_thread(generate_image_from_headline, headline)
            
            # Check if narration was paused during image generation
            if not narration_event.is_set():
                continue
                
            await safe_async_print(f"🖼️ Image saved at: {image_path}")
        except Exception as e:
            await safe_async_print(f"❌ Error generating image: {e}")

        current_index += 1

        # Wait 5 seconds with frequent checks for interruption
        for i in range(50):  # 50 * 0.1 = 5 seconds
            if not narration_event.is_set() or done.is_set():
                break
            await asyncio.sleep(0.1)

    await safe_async_print("\n✅ Narration complete.")
    done.set()

async def input_listener():
    global current_index
    
    while not done.is_set():
        try:
            # Use a timeout to periodically check if done
            user_input = await asyncio.wait_for(aioconsole.ainput(), timeout=1.0)
        except asyncio.TimeoutError:
            continue  # Check done flag and continue listening
        except Exception as e:
            await safe_async_print(f"❌ Input error: {e}")
            continue

        # Handle empty input (just Enter pressed)
        if not user_input.strip():
            if narration_event.is_set():
                narration_event.clear()
                await safe_async_print("⏸️ Narration paused. Ask a question or press Enter again to continue.")
            else:
                narration_event.set()
                await safe_async_print("▶️ Resuming narration...")
            continue

        # Handle question input
        was_running = narration_event.is_set()
        narration_event.clear()  # Pause narration

        # Build context
        if current_index == 0:
            context = f"The current headline is:\n1. {headlines[0]}"
        else:
            context = (
                "Here are the headlines narrated so far:\n" +
                "\n".join([f"{i+1}. {h}" for i, h in enumerate(headlines[:current_index])]) +
                "\nIf the user doesn't specify the headline, answer the question with the last headline."
            )

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant for news headlines. Answer based on the headlines provided."},
            {"role": "user", "content": context},
            {"role": "user", "content": user_input}
        ]

        await safe_async_print("💬 Thinking...")

        try:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            answer = response.choices[0].message.content.strip()
            await safe_async_print(f"🤖 {answer}")
            await safe_async_print("Press Enter to continue narration...")
        except Exception as e:
            await safe_async_print(f"❌ Error: {e}")

        # Auto-resume if it was running before, or wait for user input
        if was_running:
            narration_event.set()

async def main():
    safe_print("🟢 Narration starting. Press ENTER at any time to pause/resume or type a question.")
    
    # Create tasks
    narrator_task = asyncio.create_task(narrator())
    input_task = asyncio.create_task(input_listener())
    
    try:
        # Wait for narrator to complete
        await narrator_task
    except KeyboardInterrupt:
        safe_print("\n🛑 Interrupted by user")
    finally:
        # Clean shutdown
        done.set()
        narration_event.set()  # Unblock any waiting tasks
        
        # Cancel input task
        input_task.cancel()
        try:
            await input_task
        except asyncio.CancelledError:
            pass
        
        safe_print("👋 Exiting cleanly.")

# 🔁 Run the real-time session
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")