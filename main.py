"""
MCP-Integrated News Narrator System

A system that fetches live news headlines using MCP servers,
generates images for each headline, and provides an interactive narration
experience with pause/resume functionality and AI-powered Q&A.

Based on the original implementation with agents SDK.
"""

import asyncio
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import aioconsole

load_dotenv(override=True)

from diffusers import StableDiffusionPipeline
import torch

client = AsyncOpenAI()

# === Stable Diffusion ===
# Load the SD Turbo model from Hugging Face
model_id = "stabilityai/sd-turbo"

# Suppress diffusers progress bars and warnings
import logging
logging.getLogger("diffusers").setLevel(logging.ERROR)

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
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

# === MCP Integration ===
try:
    from agents import Agent, Runner, RunConfig
    from agents.mcp import MCPServerStdio
    from agents.run_context import RunContextWrapper
    from agents.tracing import trace
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️ MCP agents not available. Using fallback headlines.")

async def fetch_latest_headlines(num_headlines: int = 5) -> list[str]:
    """Fetch latest headlines using MCP server"""
    if not MCP_AVAILABLE:
        print("📰 Using fallback headlines (MCP not available)")
        fallback_headlines = [
            "Breaking: Major economic summit concludes with new trade agreements",
            "Scientists discover breakthrough in renewable energy storage technology",
            "Global climate conference addresses urgent environmental challenges", 
            "Tech industry leaders announce new AI safety initiatives",
            "International space station receives new crew members",
            "Medical researchers make progress in cancer treatment development",
            "World leaders meet to discuss cybersecurity cooperation",
            "Archaeological team uncovers ancient civilization artifacts",
            "New sustainable transportation systems launched in major cities",
            "Educational technology transforms learning in developing regions"
        ]
        return fallback_headlines[:num_headlines]

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
        async with MCPServerStdio(params=playwright_params, client_session_timeout_seconds=30) as mcp_server_browser:
            agent = Agent(
                name="investigator",
                instructions=instructions,
                model="gpt-4o-mini",
                mcp_servers=[mcp_server_browser]
            )
            # Create a tracing context
            run_context = RunContextWrapper(context={})

            with trace("investigate"):
                # Get headlines
                result = await Runner.run(
                    agent,
                    f"Find the {num_headlines} latest headlines from reliable news websites",
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
def safe_print(message):
    """Print with immediate flush to ensure visibility"""
    print(message, flush=True)

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

        # Build context - include current headline being processed
        if current_index == 0 and len(headlines) > 0:
            context = f"The current headline being processed is:\n1. {headlines[0]}"
        else:
            # Include all headlines up to and including the current one
            context_headlines = headlines[:current_index + 1] if current_index < len(headlines) else headlines[:current_index]
            context = "Headlines processed so far:\n"
            context += "\n".join([f"{i+1}. {h}" for i, h in enumerate(context_headlines)])
            
            # Add current headline being processed if it's different
            if current_index < len(headlines):
                context += f"\n\nCurrent headline being processed:\n{current_index + 1}. {headlines[current_index]}"
            
            context += "\n\nIf the user doesn't specify which headline, answer about the current headline being processed."

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
        safe_print("🟢 Narration starting. Press ENTER at any time to pause/resume or type a question.\n")
        
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