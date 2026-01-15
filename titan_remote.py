import gradio as gr
from gradio_client import Client
import shutil
import os
import time

# ==============================================================================
# TITAN REMOTE BRIDGE (V1.0)
# ==============================================================================
print("👾 TITAN REMOTE BRIDGE: LOCAL COMMAND CENTER")
print("   Connects your PC to the TITAN Cloud Core (Google Colab)")

# State
CLIENT_URL = None
client = None

def connect_to_cloud(url):
    global CLIENT_URL, client
    try:
        if not url.endswith("/"):
            url += "/"
        print(f"📡 CONNECTING TO: {url}...")
        client = Client(url)
        CLIENT_URL = url
        return f"✅ CONNECTED TO TITAN CLOUD CORE!\nURL: {url}"
    except Exception as e:
        return f"<!> CONNECTION FAILED: {e}\nCheck if the Colab Notebook is running and 'share=True'."

def local_generate_reel(action_prompt, voice_text, preset, engine_mode):
    if not client:
        yield None, "⚠️ NOT CONNECTED! Enter Colab URL above."
        return
    
    try:
        yield None, "🚀 SENDING JOB TO CLOUD... (Please wait 2-5 mins)"
        print(f"🚀 SENDING JOB TO CLOUD: {action_prompt} ({engine_mode})")
        
        # Call the API
        result = client.predict(
            action_prompt,
            voice_text,
            preset,
            engine_mode,
            api_name="/generate_reel"
        )
        
        # Result is [video_path, status_text] usually, but client returns tuple
        # Depending on Gradio version, it might be a list or tuple.
        # Gradio Client downloads the file to a temp dir locally.
        
        video_temp_path = result[0]
        status_text = result[1]
        
        # Move to local 'downloads' folder
        os.makedirs("downloads", exist_ok=True)
        local_filename = f"downloads/titan_reel_{int(time.time())}.mp4"
        
        # Check if video path is valid
        if video_temp_path and os.path.exists(video_temp_path):
            shutil.copy(video_temp_path, local_filename)
            yield local_filename, f"{status_text} (Saved to {local_filename})"
        else:
            yield None, f"⚠️ Error: No video returned. Status: {status_text}"
            
    except Exception as e:
        yield None, f"<!> API ERROR: {e}"

def local_create_character(prompt, preset, image):
    if not client:
        return None, "⚠️ NOT CONNECTED!"
    try:
        result = client.predict(
            prompt,
            preset,
            image,
            api_name="/create_character"
        )
        img_temp = result[0]
        status = result[1]
        
        os.makedirs("downloads", exist_ok=True)
        local_img = f"downloads/char_{int(time.time())}.png"
        if img_temp and os.path.exists(img_temp):
            shutil.copy(img_temp, local_img)
            return local_img, status
        return None, status
    except Exception as e:
        return None, str(e)

# ==============================================================================
# LOCAL UI (LIGHTWEIGHT)
# ==============================================================================
theme = gr.themes.Glass(primary_hue="cyan", secondary_hue="slate")

with gr.Blocks(theme=theme, title="TITAN REMOTE") as app:
    gr.Markdown("# 👾 TITAN REMOTE BRIDGE")
    gr.Markdown("### ☁️ Control the Cloud from your PC.")
    
    with gr.Row():
        colab_url_input = gr.Textbox(label="Colab Public URL (e.g. https://xxxx.gradio.live)", placeholder="Paste URL here...")
        connect_btn = gr.Button("🔗 CONNECT", variant="primary")
    
    status_box = gr.Textbox(label="Connection Status", interactive=False)
    
    with gr.Tabs():
        with gr.TabItem("🚀 Viral Reels"):
            action = gr.Textbox(label="Action Prompt")
            voice = gr.Textbox(label="Voiceover")
            preset = gr.Dropdown(["✨ TikTok Perfect (Standard)", "🦇 Goth/Alt Aesthetic", "👻 Creepypasta/Horror"], label="Preset", value="✨ TikTok Perfect (Standard)")
            engine = gr.Radio(["Action Director (AnimateDiff)", "GOD MODE (CogVideoX-5B - Sora Class)"], label="Engine", value="GOD MODE (CogVideoX-5B - Sora Class)")
            
            gen_btn = gr.Button("GENERATE (CLOUD)", variant="stop")
            out_vid = gr.Video(label="Downloaded Result")
            out_stat = gr.Label()
            
            gen_btn.click(local_generate_reel, [action, voice, preset, engine], [out_vid, out_stat])
            
    connect_btn.click(connect_to_cloud, inputs=[colab_url_input], outputs=[status_box])

if __name__ == "__main__":
    # Disable SSE queue to prevent WebSocket errors
    app.queue(api_open=False).launch(show_error=True)
