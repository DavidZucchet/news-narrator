"""Full MCP + narrator integration"""


import asyncio
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import aioconsole
import sys

# MCP Integration imports
from agents import Agent, Runner, RunConfig
from agents.mcp import MCPServerStdio
from agents.run_context import RunContextWrapper
from agents.tracing import trace

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

# === MCP Server Integration ===
async def fetch_latest_headlines(num_headlines: int = 5) -> list[str]:
    """Fetch latest headlines using MCP server"""
    playwright_params = {"command": "npx", "args": ["@playwright/mcp@latest"]}
    
    instructions = f"""
    You browse the internet to accomplish your instructions.
    You are highly capable at browsing the internet independently to accomplish your task, 
    including accepting all cookies and clicking 'not now' as
    appropriate to get to the content you need. If one website isn't fruitful, try another. 
    Be persistent until you have solved your assignment,
    trying different options and sites as needed.
    
    Please return ONLY a numbered list of {num_headlines} headlines, nothing else.
    Format each headline as: "1. [headline text]", "2. [headline text]", etc.
    """
    
    try:
        async with MCPServerStdio(params=playwright_params, client_session_timeout_seconds=60) as mcp_server_browser:
            agent = Agent(
                name="investigator",
                instructions=instructions,
                model="gpt-4o-mini",
                mcp_servers=[mcp_server_browser]
            )
            
            run_context = RunContextWrapper(context={})
            
            with trace("fetch_headlines"):
                result = await Runner.run(
                    agent,
                    f"Find the {num_headlines} latest news headlines from reliable news websites. Return only the headlines in a numbered list format.",
                    context=run_context
                )
                
                # Parse the result to extract headlines
                headlines_text = result.final_output
                headlines = []
                
                # Split by lines and extract headlines
                lines = headlines_text.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and any(line.startswith(f"{i}.") for i in range(1, num_headlines + 1)):
                        # Remove the number prefix and clean up
                        headline = line.split('.', 1)[1].strip()
                        if headline:
                            headlines.append(headline)
                
                # Fallback if parsing fails
                if not headlines:
                    # Try to extract any meaningful lines
                    for line in lines:
                        line = line.strip()
                        if line and len(line) > 10:  # Reasonable headline length
                            headlines.append(line)
                            if len(headlines) >= num_headlines:
                                break
                
                return headlines[:num_headlines] if headlines else [
                    "Unable to fetch headlines - using fallback data"
                ]
                
    except Exception as e:
        print(f"❌ Error fetching headlines: {e}")
        # Fallback headlines
        return [
            "Error fetching live headlines - using sample data",
            "MCP server connection failed",
            f"Requested {num_headlines} headlines but encountered technical issues",
            "Please check your internet connection and MCP setup",
            "Using local fallback headlines for demonstration"
        ][:num_headlines]

# Global variables
headlines = []
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

async def answer_question(question: str, context: str = "") -> str:
    """Answer questions about headlines using OpenAI"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI that explains news headlines."},
                {"role": "user", "content": f"{context}\n\n{question}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Sorry, I couldn't process your question due to an error: {e}"

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

        await safe_async_print("💬 Thinking...")

        try:
            answer = await answer_question(user_input, context)
            await safe_async_print(f"🤖 {answer}")
            await safe_async_print("Press Enter to continue narration...")
        except Exception as e:
            await safe_async_print(f"❌ Error: {e}")

        # Auto-resume if it was running before
        if was_running:
            narration_event.set()

async def get_user_headline_count():
    """Get the number of headlines from user input"""
    while True:
        try:
            safe_print("📊 How many headlines would you like to fetch? (1-20, default: 5): ")
            user_input = await aioconsole.ainput()
            
            if not user_input.strip():
                return 5  # Default
            
            count = int(user_input.strip())
            if 1 <= count <= 20:
                return count
            else:
                safe_print("❌ Please enter a number between 1 and 20.")
        except ValueError:
            safe_print("❌ Please enter a valid number.")
        except Exception as e:
            safe_print(f"❌ Error: {e}")
            return 5  # Default on error

async def main():
    global headlines, current_index
    
    safe_print("🚀 MCP-Integrated News Narrator Starting...")
    
    # Get number of headlines from user
    num_headlines = await get_user_headline_count()
    safe_print(f"📰 Fetching {num_headlines} latest headlines...")
    
    # Fetch headlines using MCP server
    try:
        headlines = await fetch_latest_headlines(num_headlines)
        safe_print(f"✅ Successfully fetched {len(headlines)} headlines!")
        
        # Display fetched headlines
        safe_print("\n📋 Headlines to be narrated:")
        for i, headline in enumerate(headlines, 1):
            safe_print(f"   {i}. {headline}")
        safe_print("")
        
    except Exception as e:
        safe_print(f"❌ Error fetching headlines: {e}")
        safe_print("Using fallback headlines...")
        headlines = [
            "Trump says Israel has agreed to conditions for 60-day Gaza ceasefire",
            "NASA plans new moon mission by 2026",
            "Global markets rally as inflation slows",
            "AI surpasses human accuracy in cancer detection",
            "Massive solar storm expected to hit Earth this week"
        ][:num_headlines]
    
    current_index = 0
    
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