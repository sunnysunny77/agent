import os
import gradio as gr
from smolagents import (
    tool,
    CodeAgent,
    DuckDuckGoSearchTool,
    InferenceClientModel,
    FinalAnswerTool,
)
from huggingface_hub import InferenceClient
from PIL import Image
import io
import base64
from fastapi import FastAPI

token = os.getenv("HF_TOKEN")

image_client = InferenceClient(
    model="stabilityai/stable-diffusion-xl-base-1.0",
    token=token
)

last_generated_image = None

@tool
def image_tool(prompt: str) -> str:
    """
    Generate an image from text.
    Args:
        prompt (str): image description
    Returns:
        str: A confirmation message.
    """
    global last_generated_image
    
    image = image_client.text_to_image(prompt)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    last_generated_image = image 
    
    return "Image has been generated successfully!"
    
@tool
def search_tool(query: str)-> str:
    """
    Search the web and return the most relevant results.

    Args:
        query (str): The search query.

    Returns:
        str: The search results.
    """
    web_search_tool = DuckDuckGoSearchTool(max_results=5, rate_limit=2.0)
    results = web_search_tool(query)
    return results

    
sentiment_client = InferenceClient(token=token)

@tool
def sentiment_tool(text: str) -> str:
    """
    Analyze sentiment of given text.

    Args:
        text (str): The sentiment query.
    
    Returns: str: sentiment
    """
    messages = [
        {"role": "system", "content": "Analyze the sentiment of the following text"},
        {"role": "user", "content": text},
    ]   
    
    completion = sentiment_client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=messages,
        max_tokens=150,
    )

    result = completion.choices[0].message.content

    return result

final_answer = FinalAnswerTool()

model = InferenceClientModel(
    model_id="meta-llama/Llama-3.3-70B-Instruct",
    token=token,
    max_tokens=2096,
    temperature=0.5,
)

agent = CodeAgent(
    model=model,
    tools=[
        image_tool,
        sentiment_tool,
        search_tool,
        final_answer,
    ],
    max_steps=6,
    planning_interval=None,
)

agent.prompt_templates["system_prompt"] = agent.prompt_templates["system_prompt"] + """"
    You are a tool calling agent.
    You have access to these tools: 
    - sentiment_tool(text: str) -> str 
    - Analyze sentiment of given text.
    - search_tool(query: str) -> str
    - Search the web and return the most relevant results.
    - Used for sentiment analysis
    - image_tool(prompt: str) -> str
    - Generate an image from text.
    - You must construct a well-formatted human-readable answer
    - You must introduce yourself as Jerry and greet the user in the answer
    - You must try include newlines, bullets, numbering, and proper punctuation
    - You must use this answer in final_answer
"""

def run_agent(query: str):
    global last_generated_image
    last_generated_image = None
    agent_text_response = agent.run(query)

    return agent_text_response, last_generated_image

iface = gr.Interface(
    fn=run_agent,
    inputs=gr.Textbox(lines=8),
    outputs=[
        gr.Textbox(label="Agent Response", lines=8),
        gr.Image(label="Generated Image"),
    ],
    title="SmolAgent",
    description="Search • Sentiment • Image Generation",
)

# Create the FastAPI instance
app = FastAPI()

# Mount Gradio into FastAPI
# "path='/'" is crucial here to prevent redirect loops to /index.html
app = gr.mount_gradio_app(app, iface, path="/")