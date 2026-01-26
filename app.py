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
import concurrent.futures

def run_with_timeout(func, timeout=300, *args, **kwargs):
    """Run function with timeout"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Tool timed out after {timeout} seconds")

def pil_to_tempfile(image):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_path = tmp.name
    tmp.close()
    image.save(tmp_path, format="PNG")
    return tmp_path

token = os.getenv("HF_TOKEN")

client = InferenceClient(token=token)

video_client = InferenceClient(
    model="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    provider="fal-ai",
    api_key=token,
)

nsfw_image_detection_client = InferenceClient(
    provider="hf-inference",
    api_key=token
)

text_to_image_client = InferenceClient(
    model="stabilityai/stable-diffusion-3-medium",
    api_key=token
)

def resize_and_crop(image, target_res=(832, 480)):
    tw, th = target_res
    iw, ih = image.size
    scale = max(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    image = image.resize((nw, nh), Image.LANCZOS)
    left = (nw - tw) // 2
    if ih > iw:
        top = int((nh - th) * 0.25)
    else:
        top = (nh - th) // 2
    right = left + tw
    bottom = top + th
    return image.crop((left, top, right, bottom))

def aligned_num_frames(duration, fps=16):
    n = int(duration * fps)
    return ((n - 1) // 4) * 4 + 1

image_output = None    
video_output = None    

video_prompt = ""
video_duration = 4
video_steps = 20
video_guidance = 3.0
video_randomize = True

@tool
def video_tool(
    video_image_input: Image.Image, 
    prompt: str = "high quality, detailed, sharp, cinematic", 
    duration: float = 4, 
    steps: int = 20, 
    guidance: float = 3.0
) -> str:
    """
    Generates a video from a starting image using Wan 2.1.
    Args:
        video_image_input (Image.Image): The source image to be animated.
        prompt (str): The prompt for video generation.
        duration (float): Duration in seconds.
        steps (int): Number of inference steps.
        guidance (float): Guidance scale.
    Returns:
        str: A confirmation message.
    """
    global video_output
    
    try:
        FPS = 12  
        num_frames = aligned_num_frames(duration, FPS) 
        
        def generate_video():
            return video_client.image_to_video(
                image=video_image_input.resize((832, 480)),
                prompt=prompt,
                negative_prompt="low quality, deformed, grainy, blurry, pixelated",
                num_frames=num_frames,
                num_inference_steps=steps,  
                guidance_scale=guidance, 
                decode_chunk_size=8, 
            )
        
        video_bytes = run_with_timeout(generate_video, timeout=300)
        
        out = tempfile.mktemp(suffix=".mp4")
        with open(out, "wb") as f:
            f.write(video_bytes)
        
        video_output = out
        return "Video successfully generated and stored for Gradio UI."
        
    except Exception as e:
        video_output = None
        return f"Video generation failed: {e}"
      
@tool
def nsfw_detection_tool(nsfw_detection_input: Image.Image) -> str:
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

@tool
def image_tool(prompt: str) -> str:
    """
    Generate an image from text using SD3-Medium.
    Args:
        prompt (str): image description.
    Returns:
        str: A confirmation message.
    """    
    global image_output  
    
    try:
        def generate_image():
            return text_to_image_client.text_to_image(
                prompt=prompt,
                negative_prompt="low quality, deformed",
                guidance_scale=7.0,
                num_inference_steps=28,
                width=992,
                height=992
            )
        
        image = run_with_timeout(generate_image, timeout=300)
        image_output = image
        return "Image successfully generated and stored for Gradio UI."
        
    except Exception as e:
        image_output = None
        return f"Image generation failed: {e}"


@tool
def search_tool(query: str) -> str:
    """
    Search the web and return the most relevant results.
    Args:
        query (str): The search query.
    Returns:
        str: The search results.
    """
    try:
        web_search_tool = DuckDuckGoSearchTool(max_results=5, rate_limit=2.0)
        results = web_search_tool(query)
        return results
        
    except Exception as e:
        return f"Search failed: {e}" 

final_answer = FinalAnswerTool()

model = InferenceClientModel(
    model_id="meta-llama/Llama-3.3-70B-Instruct",
    token=token,
    max_tokens=2096,
    temperature=0.6,
)

agent = CodeAgent(
    model=model,
    tools=[
        video_tool,
        image_tool,
        nsfw_detection_tool,
        search_tool,
        final_answer,
    ],
    max_steps=6,
    planning_interval=None,
)

agent.prompt_templates["system_prompt"] += """
    You are a tool calling agent.
    You have access to these tools: 
    - search_tool(query: str) -> str
    - Search the web and return the most relevant results.
    - Used for sentiment analysis
    - video_tool(video_image_input: Image.Image, prompt: str, duration: float, steps: int, guidance: float) -> str
    - Generate a video from an image input with custom parameters, if successfull or not you will be notified by the return string.    
    - image_tool(prompt: str) -> str
    - Generate an image from a text prompt, if successfull or not you will be notified by the return string.
    - nsfw_detection_tool(nsfw_detection_input: Image.Image) -> str
    - The nsfw_detection_input additional argument is processed entirely within the tool to produce a score from the input.
    - When sentiment analysis is requested, you must analyze the sentiment of prompt text using a range score of 0 -> 10 
    - and provied alternative wording.
    - When generating a video, to save time the image must not use the nsfw_detection_tool first.
    - You must construct a well-formatted human-readable answer
    - You must introduce yourself as Jerry and greet the user in the answer
    - You must try include newlines, bullets, numbering, and proper punctuation
    - You must use this answer in final_answer
"""

def run_agent(
    query, 
    nsfw_detection_input, 
    video_image_input, 
    video_prompt_param="", 
    video_duration_param=4.0, 
    video_steps_param=20, 
    video_guidance_param=3.0, 
    video_randomize_param=True
):
    global image_output, video_output
    image_output = None
    video_output = None

    yield None, None, "⏳ Jerry is thinking... please wait"

    try:

        actual_query = ""
        
        if query and query.strip():
            actual_query = query
        elif video_image_input is not None:
            actual_query = f"Generate a video from this image with prompt: '{video_prompt_param}', duration: {video_duration_param}s, steps: {video_steps_param}, guidance: {video_guidance_param}"
        elif nsfw_detection_input is not None:
            actual_query = "Check this image for NSFW content"
        else:
            actual_query = "What can I help you with?"

        response = agent.run(
            actual_query,
            additional_args={
                "nsfw_detection_input": nsfw_detection_input, 
                "video_image_input": video_image_input,
                "prompt": video_prompt_param,
                "duration": video_duration_param,
                "steps": video_steps_param,
                "guidance": video_guidance_param,
            }
        )

        yield image_output, video_output, str(response)

    except Exception as e:
        yield None, None, f"❌ Agent Error: {str(e)}"

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
            )
            
        with gr.Row():
            run_chat_btn = gr.Button("🚀 Run", variant="primary")

        gr.Examples(
            examples=[
                "How do i cook a curry quickly",
                "Analyze the sentiment: This is terrible service",
                "Translate this text to English 在线中文输入",
            ],
            inputs=[query_chat],
            label="💡 Try these:"
        )

        run_chat_btn.click(
            fn=run_agent,
            inputs=[
                query_chat,
                gr.Image(visible=False),
                gr.Image(visible=False),
                gr.Textbox(visible=False),
                gr.Number(visible=False),
                gr.Number(visible=False),
                gr.Number(visible=False),
                gr.Checkbox(visible=False),
            ],
            outputs=[gr.Image(visible=False), gr.Video(visible=False), agent_response]
        )

    with gr.Tab("🎬 Video Tools"):
        with gr.Row():
            with gr.Column():
                video_image_input = gr.Image(type="pil", label="Input Image")
                gr.Markdown("Upload the starting image for the video. This will be animated according to your prompt.")
    
                prompt_txt = gr.Textbox(lines=3, label="Prompt", placeholder="Video diffusion may take awhile..")
                gr.Markdown("Describe what you want in the video. Be as detailed as needed.")
    
                with gr.Accordion("Settings", open=True):
                    dur_slider = gr.Slider(1, 4, value=4, step=0.1, label="Duration (seconds)")
                    gr.Markdown("Controls the length of the video. Longer durations generate more frames and require more compute.")
                    
                    step_slider = gr.Slider(4, 20, value=20, step=1, label="Steps (quality)")
                    gr.Markdown("Number of diffusion steps used per frame. Higher values improve detail and temporal stability but increase generation time.")
                    
                    guidance_slider = gr.Slider(1.0, 6.0, value=3.0, step=0.1, label="Guidance Strength")
                    gr.Markdown("How strongly the model follows the prompt. Lower values are more natural and fluid, higher values are more literal and stylized.")
    
                    rand_check = gr.Checkbox(value=True, label="Randomize Seed")
                    gr.Markdown("Toggle this to get a fresh variation every run. Turn it off to reuse the same motion and style.")
    
                gen_btn = gr.Button("Generate Video", variant="primary")

            with gr.Column():
                output_vid = gr.Video(label="Generated Video")

        gr.Examples(
            examples=[
                "The people all raise a glass and cheer",
            ],
            inputs=[prompt_txt],
            label="💡 Try these:"
        )
        
        gen_btn.click(
            fn=run_agent,
            inputs=[
                gr.Textbox(visible=False), 
                gr.Image(visible=False),    
                video_image_input,            
                prompt_txt,                 
                dur_slider,                  
                step_slider,              
                guidance_slider,               
                rand_check,                    
            ],
            outputs=[gr.Image(visible=False), output_vid, agent_response]
        )

    with gr.Tab("🎨 Image Tools"):
        with gr.Row():
            with gr.Column():
                nsfw_detection_input = gr.Image(type="pil", label="Upload for NSFW Check")
                check_nsfw_btn = gr.Button("🔍 Check NSFW")
                query_img = gr.Textbox(lines=2, label="Image generation prompt")
                run_img_btn = gr.Button("🎨 Generate Image", variant="primary")

            with gr.Column():
                image_output_display = gr.Image(label="Generated Image")

        gr.Examples(
            examples=[
                "A cyberpunk cat with neon glowing eyes",
                "A serene Japanese garden with cherry blossoms",
                "A futuristic city with flying cars at sunset",
                "A magical forest with bioluminescent plants",
                "A steampunk robot drinking tea in a Victorian parlor"
            ],
            inputs=[query_img],
            label="💡 Try these:"
        )

        check_nsfw_btn.click(
            fn=run_agent,
            inputs=[
                gr.Textbox(visible=False),  
                nsfw_detection_input,         
                gr.Image(visible=False),    
                gr.Textbox(visible=False),  
                gr.Number(visible=False),    
                gr.Number(visible=False),    
                gr.Number(visible=False),    
                gr.Checkbox(visible=False), 
            ],
            outputs=[gr.Image(visible=False), gr.Video(visible=False), agent_response]
        )

        run_img_btn.click(
            fn=run_agent,
            inputs=[
                query_img,                    
                gr.Image(visible=False),      
                gr.Image(visible=False),      
                gr.Textbox(visible=False),   
                gr.Number(visible=False),     
                gr.Number(visible=False),     
                gr.Number(visible=False),      
                gr.Checkbox(visible=False),   
            ],
            outputs=[image_output_display, gr.Video(visible=False), agent_response]
        )

app = FastAPI()

app = gr.mount_gradio_app(app, demo, path="/", theme=gr.themes.Soft())