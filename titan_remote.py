import gradio as gr
from gradio_client import Client
import shutil
import os
import time

# ==============================================================================
# TITAN REMOTE BRIDGE (V2.0 - FULL CONTROL)
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

def check_connection():
    if not client:
        return False
    return True

# --- API WRAPPERS ---

def local_create_character(prompt, preset, image):
    if not check_connection():
        yield None, "⚠️ NOT CONNECTED! Enter Colab URL above."
        return
    
    try:
        yield None, "📸 GENERATING CHARACTER... (Sending to Cloud)"
        
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
            yield local_img, f"{status} (Saved locally)"
        else:
            yield None, f"⚠️ Error: No image returned. {status}"
            
    except Exception as e:
        yield None, f"<!> API ERROR: {e}"

def local_generate_reel(action_prompt, voice_text, preset, engine_mode):
    if not check_connection():
        yield None, "⚠️ NOT CONNECTED! Enter Colab URL above."
        return
    
    try:
        yield None, "🚀 SENDING JOB TO CLOUD... (Please wait 2-5 mins)"
        print(f"🚀 SENDING JOB TO CLOUD: {action_prompt} ({engine_mode})")
        
        result = client.predict(
            action_prompt,
            voice_text,
            preset,
            engine_mode,
            api_name="/generate_reel"
        )
        
        video_temp_path = result[0]
        status_text = result[1]
        
        os.makedirs("downloads", exist_ok=True)
        local_filename = f"downloads/titan_reel_{int(time.time())}.mp4"
        
        if video_temp_path and os.path.exists(video_temp_path):
            shutil.copy(video_temp_path, local_filename)
            yield local_filename, f"{status_text} (Saved locally)"
        else:
            yield None, f"⚠️ Error: No video returned. Status: {status_text}"
            
    except Exception as e:
        yield None, f"<!> API ERROR: {e}"

def local_generate_horror(story_script, scene_desc):
    if not check_connection():
        yield None, "⚠️ NOT CONNECTED! Enter Colab URL above."
        return

    try:
        yield None, "👻 GENERATING HORROR STORY... (This may take 5-10 mins)"
        
        result = client.predict(
            story_script,
            scene_desc,
            api_name="/generate_horror"
        )
        
        video_temp_path = result[0]
        status_text = result[1]
        
        os.makedirs("downloads", exist_ok=True)
        local_filename = f"downloads/horror_{int(time.time())}.mp4"
        
        if video_temp_path and os.path.exists(video_temp_path):
            shutil.copy(video_temp_path, local_filename)
            yield local_filename, f"{status_text} (Saved locally)"
        else:
            yield None, f"⚠️ Error: No video returned. {status_text}"

    except Exception as e:
        yield None, f"<!> API ERROR: {e}"

# ==============================================================================
# LOCAL UI (FULL MIRROR)
# ==============================================================================
theme = gr.themes.Glass(primary_hue="cyan", secondary_hue="slate")

def open_output_folder():
    path = os.path.abspath("downloads")
    os.makedirs(path, exist_ok=True)
    os.startfile(path)
    return f"📂 Opened: {path}"

with gr.Blocks(theme=theme, title="TITAN REMOTE V2") as app:
    gr.Markdown("# 👾 TITAN REMOTE: LOCAL CONTROL CENTER")
    gr.Markdown("### ☁️ Powered by Google Colab Cloud Core")
    
    with gr.Row():
        colab_url_input = gr.Textbox(label="Colab Public URL (e.g. https://xxxx.gradio.live)", placeholder="Paste URL from Colab here...")
        connect_btn = gr.Button("🔗 CONNECT TO CORE", variant="primary")
        folder_btn = gr.Button("📂 OPEN DOWNLOADS FOLDER", variant="secondary")
    
    status_box = gr.Textbox(label="System Status", interactive=False)
    
    folder_btn.click(open_output_folder, inputs=[], outputs=[status_box])
    
    with gr.Tabs():
        # TAB 1: IDENTITY
        with gr.TabItem("👤 Identity Station"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Create New Character")
                    preset_dd = gr.Dropdown(
                        ["None", "✨ TikTok Perfect (Standard)", "🖤 TikTok Artistic (B&W)", "🦋 Unique Beauty (Vitiligo)", 
                         "🦌 Natural Beauty (Freckles)", "🦇 Goth/Alt Aesthetic", "💄 Insta Baddie (Glam)", 
                         "💪 Fitness/GymTok", "👻 Creepypasta/Horror", "🏚️ Liminal Spaces (Backrooms)"],
                        label="Style Preset", value="✨ TikTok Perfect (Standard)"
                    )
                    char_prompt = gr.Textbox(label="Character Description", placeholder="e.g. Blonde woman, blue eyes, red dress...")
                    char_upload = gr.Image(label="Or Upload Source Image", type="filepath")
                    create_char_btn = gr.Button("📸 GENERATE IDENTITY", variant="primary")
                    
                with gr.Column():
                    char_preview = gr.Image(label="Identity Preview")
                    char_status = gr.Label()
            
            create_char_btn.click(local_create_character, [char_prompt, preset_dd, char_upload], [char_preview, char_status])

        # TAB 2: VIRAL REELS
        with gr.TabItem("🚀 Viral Reels"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Generate Content")
                    action = gr.Textbox(label="Action Prompt", placeholder="Dancing to techno, pointing at text...")
                    voice = gr.Textbox(label="Voiceover Text (Optional)", placeholder="Did you know...")
                    preset_reel = gr.Dropdown(
                        ["✨ TikTok Perfect (Standard)", "🦇 Goth/Alt Aesthetic", "👻 Creepypasta/Horror"], 
                        label="Preset", value="✨ TikTok Perfect (Standard)"
                    )
                    engine = gr.Radio(
                        ["Action Director (AnimateDiff)", "Realism Engine (SVD-XT Pro)", "GOD MODE (CogVideoX-5B - Sora Class)"], 
                        label="Engine", value="GOD MODE (CogVideoX-5B - Sora Class)"
                    )
                    
                    gen_reel_btn = gr.Button("🎬 GENERATE REEL", variant="stop")
                    
                with gr.Column():
                    out_vid = gr.Video(label="Downloaded Result")
                    out_stat = gr.Label()
            
            gen_reel_btn.click(local_generate_reel, [action, voice, preset_reel, engine], [out_vid, out_stat])

        # TAB 3: HORROR
        with gr.TabItem("👻 Horror Studio"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Generate Horror Story")
                    story = gr.Textbox(label="Script (One line per scene)", lines=5)
                    scenes = gr.Textbox(label="Visuals (One line per scene)", lines=5)
                    gen_horror_btn = gr.Button("💀 GENERATE HORROR", variant="stop")
                
                with gr.Column():
                    horror_vid = gr.Video(label="Horror Result")
                    horror_stat = gr.Label()
            
            gen_horror_btn.click(local_generate_horror, [story, scenes], [horror_vid, horror_stat])

    connect_btn.click(connect_to_cloud, inputs=[colab_url_input], outputs=[status_box])

if __name__ == "__main__":
    # Disable SSE queue to prevent WebSocket errors
    app.queue(api_open=False).launch(show_error=True)
