import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load environment variables from the local .env file
load_dotenv(override=True)

async def main():
    # 1. Initialize the OpenAI model client
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    # 2. Create the AssistantAgent with the exact system message required
    agent = AssistantAgent(
        name="hr_assistant",
        model_client=model_client,
        system_message="You are an HR assistant."
    )

    # 3. Define the prompt requesting exactly 5 bullet points
    prompt = (
        "Write a 5-line job description for a Junior Python Developer. "
        "Provide exactly 5 numbered lines, one sentence per line, with no extra intro or outro text."
    )

    # 4. Stream the output to the terminal
    await Console(agent.run_stream(task=prompt))

    # 5. Close the client to avoid hanging the process
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())