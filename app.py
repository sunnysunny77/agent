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
from fastapi import FastAPI
import tempfile
from PIL import Image

def pil_to_tempfile(image):
   
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_path = tmp.name
    tmp.close()

    image.save(tmp_path, format="PNG")

    return tmp_path

token = os.getenv("HF_TOKEN")

client = InferenceClient(token=token)

nsfw_image_detection_client = InferenceClient(
    provider="hf-inference",
    api_key=token
)

text_to_image_client = InferenceClient(
    model="stabilityai/stable-diffusion-3-medium",
    api_key=token
)

@tool
def nsfw_detection_tool(nsfw_detection_input:  Image.Image) -> str:
    """
    Suitable for filtering through score explicit or inappropriate content in images.
    Args:
        nsfw_detection_input (Image.Image): The image to check.
    Returns:
        str: Highest score result.
    """ 
    try:
  
        tmp_path = pil_to_tempfile(nsfw_detection_input)
        
        outputs = client.image_classification(
            tmp_path,
            model="Falconsai/nsfw_image_detection"
        )
        
        os.remove(tmp_path)

        top_result = max(outputs, key=lambda x: x.score)

        verdict = (
            f"Verdict: {top_result.label.upper()}\n"
            f"Confidence: {top_result.score:.2%}"
        )

        return verdict

    except Exception as e:
        return f"NSFW detection failed: {e}"

image_output = None

@tool
def image_tool(prompt: str) -> str:
    """
    Generate an image from text using SD3-Medium.
    Args:
        prompt (str): image description
    Returns:
        str: A confirmation message.
    """    
    global image_output
    
    try:
        image = text_to_image_client.text_to_image(
            prompt=prompt,
            negative_prompt="blurry, distorted, low quality",
            guidance_scale=7.0,
            num_inference_steps=28,
            width=1024,
            height=1024
        )
        image_output = image
        return "Image successfully generated and stored for Gradio UI."
        
    except Exception as e:
        image_output = None
        print(f"Image generation failed: {e}")
        return f"Image generation failed: {e}"
   
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
    
    completion = client.chat.completions.create(
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
        nsfw_detection_tool,
        sentiment_tool,
        search_tool,
        final_answer,
    ],
    max_steps=6,
    planning_interval=None,
)

agent.prompt_templates["system_prompt"] += """
    You are a tool calling agent.
    You have access to these tools: 
    - sentiment_tool(text: str) -> str 
    - Analyze sentiment of given text.
    - search_tool(query: str) -> str
    - Search the web and return the most relevant results.
    - Used for sentiment analysis
    - image_tool(prompt: str) -> str
    - Generate an image from a text prompt, if successfull or not you will be notified by the return string.
    - nsfw_detection_tool(nsfw_detection_input: Image.Image) -> str
    - The nsfw_detection_input additional argument is processed entirely within the tool to produce a score from the input.
    - You must construct a well-formatted human-readable answer
    - You must introduce yourself as Jerry and greet the user in the answer
    - You must try include newlines, bullets, numbering, and proper punctuation
    - You must use this answer in final_answer
"""

def run_agent(query, nsfw_detection_input):
    global image_output
    image_output = None
    
    try:
        response = agent.run(
            query if query else "", 
            additional_args={"nsfw_detection_input": nsfw_detection_input}
        )
        return image_output, str(response)
    except Exception as e:
        return None, f"Agent Error: {str(e)}"

with gr.Blocks(title="Jerry AI Assistant") as demo:
    gr.Markdown("# 🤖 Jerry - Your AI Assistant")
    
    agent_response = gr.Textbox(
        label="Response",
        lines=5,
        interactive=False
    )
    
    with gr.Tab("💬 Chat"):
        with gr.Row():
            query_chat = gr.Textbox(
                lines=3, 
                label="Ask me anything...",
                placeholder="Generate an image of a cat, analyze its sentiment, etc.",
                scale=4
            )
        
        with gr.Row():
            run_chat_btn = gr.Button("🚀 Run", variant="primary", scale=1)
            clear_chat_btn = gr.Button("🗑️ Clear", scale=0)
        
        gr.Examples(
            examples=[
                "How do i cook a curry quickly",
                "Analyze the sentiment: This is terrible service",
            ],
            inputs=[query_chat],
            label="💡 Try these:"
        )
        
        # Hidden components for chat tab
        hidden_image_chat = gr.Image(visible=False)
        
        run_chat_btn.click(
            fn=run_agent, 
            inputs=[query_chat, hidden_image_chat],
            outputs=[hidden_image_chat, agent_response]
        )
    
    with gr.Tab("🎨 Image Tools"):
        with gr.Row():
            nsfw_detection_input = gr.Image(
                label="Upload for NSFW check",
                type="pil",
                height=300
            )
            image_output = gr.Image(
                label="Generated Image",
                height=300
            )
        
        with gr.Row():
            query_img = gr.Textbox(
                lines=2,
                label="Image generation prompt",
                placeholder="A beautiful sunset over mountains..."
            )
        
        with gr.Row():
            run_img_btn = gr.Button("🎨 Generate Image", variant="primary")
            check_nsfw_btn = gr.Button("🔍 Check NSFW")
        
        gr.Examples(
            examples=[
                "A cyberpunk cat with neon glowing eyes",
                "A serene Japanese garden with cherry blossoms",
                "A futuristic city with flying cars at sunset",
                "A magical forest with bioluminescent plants",
                "A steampunk robot drinking tea in a Victorian parlor"
            ],
            inputs=[query_img],
            label="🎨 Try these prompts:"
        )
        
        hidden_text_img = gr.Textbox(visible=False)
        hidden_image_img = gr.Image(visible=False)
        
        run_img_btn.click(
            fn=run_agent,
            inputs=[query_img, hidden_image_img],
            outputs=[image_output, agent_response]
        )
        
        check_nsfw_btn.click(
            fn=run_agent,
            inputs=[hidden_text_img, nsfw_detection_input],
            outputs=[hidden_image_img, agent_response]
        )

app = FastAPI()

app = gr.mount_gradio_app(app, demo, path="/", theme=gr.themes.Soft())