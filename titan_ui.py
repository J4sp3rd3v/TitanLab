import gradio as gr
import os
from titan_orchestrator import TitanOrchestrator

# Initialize Orchestrator
titan = TitanOrchestrator()

def fn_create_character(prompt, preset, image_input):
    if image_input is not None:
        path = f"char_upload_{os.urandom(4).hex()}.png"
        image_input.save(path)
        titan.director.set_character(path)
        return path, f"✅ Character Loaded: {path}"
    
    if prompt or (preset and preset != "None"):
        full_prompt = titan.director.apply_preset(prompt, preset)
        path, _ = titan.director.create_character(full_prompt)
        return path, f"✅ Character Generated: {path} (Preset: {preset})"
        
    return None, "⚠️ Please provide a prompt or upload an image."

def fn_generate_viral_reel(action_prompt, voice_text, preset, engine_mode):
    if not titan.director.current_character_path:
        return None, "⚠️ No Character Locked!"
        
    video_path = titan.generate_viral_reel(
        character_prompt="", # Already locked or implicit
        action_prompt=action_prompt,
        voice_text=voice_text,
        preset=preset,
        engine=engine_mode
    )
    return video_path, "✅ Viral Reel Created!"

def fn_generate_motion_transfer(prompt, source_video):
    if not titan.director.current_character_path:
        return None, "⚠️ No Character Locked!"
    if source_video is None:
        return None, "⚠️ No Source Video Uploaded!"
        
    video_path = titan.director.generate_motion_transfer(prompt, source_video)
    return video_path, "✅ Motion Transfer Complete!"

def fn_generate_auto_movie(prompt):
    video_clips, audio_clips = titan.generate_full_movie_auto(prompt)
    # Return first clip for now
    return video_clips[0], f"✅ Generated {len(video_clips)} movie clips."

# ==============================================================================
# UI LAYOUT
# ==============================================================================
theme = gr.themes.Soft(primary_hue="cyan", secondary_hue="slate", neutral_hue="slate").set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_width="1px",
    block_border_color="*primary_800"
)

with gr.Blocks(theme=theme, title="TITAN VIRAL STUDIO") as app:
    gr.Markdown("# ⚡ TITAN: VIRAL CONTENT ENGINE")
    
    with gr.Tabs():
        # TAB 1: VIRAL REELS
        with gr.TabItem("📱 Viral Reels"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 1. Identity")
                    preset_dd = gr.Dropdown(titan.director.get_preset_list(), label="Style Preset", value="✨ TikTok Perfect (Standard)")
                    char_prompt = gr.Textbox(label="Character Details", placeholder="Blonde, blue eyes...")
                    char_upload = gr.Image(label="Or Upload Image", type="pil")
                    create_btn = gr.Button("Set Identity")
                    char_status = gr.Label()
                    
                    gr.Markdown("### 2. Content")
                    action_prompt = gr.Textbox(label="Action/Dance", placeholder="Dancing to techno, pointing at text...")
                    voice_text = gr.Textbox(label="Voiceover (Optional)", placeholder="Did you know this fact?...")
                    engine_mode = gr.Radio(
                        ["Action Director (AnimateDiff)", "Realism Engine (SVD-XT Pro)", "PixVerse Cloud (Requires Key)", "GOD MODE (CogVideoX-5B - Sora Class)"], 
                        label="Generation Engine", 
                        value="Action Director (AnimateDiff)",
                        info="Action = Dancing/Long Video | Realism = Photorealistic/Short | GOD MODE = Best Open Source Model (10GB+)"
                    )
                    generate_reel_btn = gr.Button("🚀 GENERATE REEL", variant="primary")
                    
                with gr.Column():
                    preview_img = gr.Image(label="Identity Preview", height=300)
                    result_video = gr.Video(label="Final Reel")
                    
            create_btn.click(fn_create_character, [char_prompt, preset_dd, char_upload], [preview_img, char_status], api_name="create_character")
            generate_reel_btn.click(fn_generate_viral_reel, [action_prompt, voice_text, preset_dd, engine_mode], [result_video, char_status], api_name="generate_reel")

        # TAB 2: MOTION TRANSFER (NANO BANANA STYLE)
        with gr.TabItem("💃 Motion Transfer (Nano Style)"):
             with gr.Row():
                with gr.Column():
                    gr.Markdown("### Source Motion")
                    motion_prompt = gr.Textbox(label="Description", placeholder="A robot dancing...")
                    source_vid = gr.Video(label="Upload Source Video (Dance/Move)")
                    gen_motion_btn = gr.Button("🍌 GENERATE MOTION TRANSFER", variant="primary")
                with gr.Column():
                    motion_result = gr.Video(label="Result")
                    motion_status = gr.Label()
             gen_motion_btn.click(fn_generate_motion_transfer, [motion_prompt, source_vid], [motion_result, motion_status], api_name="generate_motion_transfer")

        # TAB 3: AUTO MOVIE
        with gr.TabItem("🎬 Auto Movie Director"):
            with gr.Row():
                with gr.Column():
                    movie_prompt = gr.Textbox(label="Movie Concept", placeholder="A cyberpunk detective story in neo-tokyo...")
                    gen_movie_btn = gr.Button("🎥 ACTION!", variant="stop")
                with gr.Column():
                    movie_result = gr.Video(label="Movie Result")
                    movie_status = gr.Label()
            gen_movie_btn.click(fn_generate_auto_movie, [movie_prompt], [movie_result, movie_status], api_name="generate_auto_movie")


if __name__ == "__main__":
    app.launch(share=True)
