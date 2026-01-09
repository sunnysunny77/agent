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
import io
from fastapi import FastAPI

token = os.getenv("HF_TOKEN")

image_client = InferenceClient(
    model="stabilityai/stable-diffusion-3-medium",
    token=token
)

last_generated_image = None

@tool
def image_tool(prompt: str) -> str:
    """
    Generate an image from text using SD3-Medium.
    Args:
        prompt (str): image description
    Returns:
        str: A confirmation message.
    """
    global last_generated_image

    image = image_client.text_to_image(
        prompt=prompt,
        negative_prompt="blurry, distorted, low quality",
        guidance_scale=7.0,
        num_inference_steps=28,
        width=1024,
        height=1024
    )

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
        {"role": "system", "content": "Analyze the sentiment of the following text using a range score of 0 -> 10 and provied alternative wording"},
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
    - Generate an image from text which will be provided to user trough the gloabl last_generated_image and gardio ui.
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

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🤖 SmolAgent — Jerry
        **Search • Sentiment • Image Generation**
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            query_box = gr.Textbox(
                lines=8,
                label="Your Query",
                placeholder="Ask me to search, analyze sentiment, or generate an image…"
            )

            run_btn = gr.Button("Run Agent", variant="primary")

        with gr.Column(scale=1):
            response_box = gr.Textbox(
                label="Agent Response",
                lines=10
            )

            image_output = gr.Image(
                label="Generated Image"
            )

    run_btn.click(
        fn=run_agent,
        inputs=query_box,
        outputs=[response_box, image_output],
    )

app = FastAPI()

app = gr.mount_gradio_app(app, demo, path="/", theme=gr.themes.Soft())