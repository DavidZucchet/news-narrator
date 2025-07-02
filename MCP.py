
""" Basic web browsing integration"""

# The imports
import asyncio
import os
from dotenv import load_dotenv
from agents import Agent, Runner, trace
from agents.mcp import MCPServerStdio
from agents.run_context import RunContextWrapper
from agents.tracing import trace

async def main():
    # Load environment variables
    load_dotenv(override=True)

    instructions = """
    You browse the internet to accomplish your instructions.
    You are highly capable at browsing the internet independently to accomplish your task, 
    including accepting all cookies and clicking 'not now' as
    appropriate to get to the content you need. If one website isn't fruitful, try another. 
    Be persistent until you have solved your assignment,
    trying different options and sites as needed.
    """
    
    playwright_params = {"command": "npx", "args": ["@playwright/mcp@latest"]}


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
            # 🧠 First turn: Get headlines
            result1 = await Runner.run(
                agent,
                "Find the 5 latest headlines from the Web",
                context=run_context
            )
            print(result1.final_output)

            # 🧠 Second turn: Retain conversation history
            inputs = result1.to_input_list()
            inputs.append({"role": "user", "content": "Can you give me more detail about the second headline?"})
            result2 = await Runner.run(
                agent,
                inputs,
                context=run_context
            )
            print(result2.final_output)


if __name__ == "__main__":
    asyncio.run(main())