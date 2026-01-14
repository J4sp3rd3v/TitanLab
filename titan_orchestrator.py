from titan_video_engine import TitanDirector
from titan_audio_engine import TitanVoice, run_tts
from titan_editor import TitanEditor
import os
import asyncio

class TitanOrchestrator:
    def __init__(self):
        self.director = TitanDirector()
        self.voice = TitanVoice()
        self.editor = TitanEditor()

    def generate_horror_story(self, story_script, scene_descriptions, voice="Male Deep (Horror)"):
        """
        Generates a full horror video from a script.
        story_script: List of text segments for narration.
        scene_descriptions: List of visual prompts matching segments.
        """
        print("👻 STARTING HORROR PRODUCTION...")
        
        video_clips = []
        audio_clips = []
        
        # 1. Generate Audio & Video for each scene
        for i, (text, prompt) in enumerate(zip(story_script, scene_descriptions)):
            print(f"   [Scene {i+1}] Processing...")
            
            # Audio
            audio_path = f"audio_scene_{i}.mp3"
            run_tts(text, voice, audio_path)
            audio_clips.append(os.path.abspath(f"audio_assets/{audio_path}"))
            
            # Video (Using SVD or Action Engine)
            # For horror, we usually want atmospheric scenes (SVD) or slow movement
            self.director.set_quality("CINEMA")
            
            # Generate Base Image
            scene_img = self.director.build_scene(prompt + ", horror style, cinematic lighting")
            
            # Animate (SVD is better for atmosphere)
            video_path = self.director.generate_realism(motion_bucket_id=80) # Slower motion for horror
            video_clips.append(os.path.abspath(video_path))
            
        # 2. Stitch together
        # This is complex because we have multiple audio/video pairs.
        # TitanEditor needs an update to handle list of pairs, or we do it here manually via Editor
        # For now, let's assume we want one long video.
        
        # TODO: Update TitanEditor to handle multi-clip timeline
        print("⚠️  Stitching not fully implemented for multi-scene yet. Returning assets.")
        return video_clips, audio_clips

    def generate_viral_reel(self, character_prompt, action_prompt, voice_text=None, preset="✨ TikTok Perfect (Standard)", engine="Action"):
        """
        Creates a single viral reel with a consistent character.
        """
        print(f"✨ CREATING VIRAL REEL: {preset} (Engine: {engine})")
        
        # 1. Character (Only needed for native engines)
        if "PixVerse" not in engine:
            self.director.create_character(self.director.apply_preset(character_prompt, preset))
        
        # 2. Action/Video
        video_path = None
        if "PixVerse" in engine:
             # Use PixVerse Bridge
             full_prompt = self.director.apply_preset(action_prompt, preset)
             video_path = self.director.pixverse.generate(full_prompt)
        elif "Realism" in engine:
             video_path = self.director.generate_realism()
        elif "GOD MODE" in engine:
             full_prompt = self.director.apply_preset(action_prompt, preset)
             video_path = self.director.generate_god_mode(full_prompt)
        else:
             # Default Action
             video_path = self.director.generate_action(action_prompt, duration_sec=4, motion_strength=6, upscale=True)
        
        if not video_path:
            return None
        
        # 3. Audio (Optional)
        audio_path = None
        if voice_text:
            voice_key = "Female Energetic (Viral)" if "Girl" in character_prompt or "Woman" in character_prompt else "Male Energetic (Viral)"
            run_tts(voice_text, voice_key, "viral_voice.mp3")
            audio_path = os.path.abspath("audio_assets/viral_voice.mp3")
            
        # 4. Edit
        if audio_path:
            final_path = self.editor.create_viral_short([video_path], audio_path)
            return final_path
        else:
            return video_path

if __name__ == "__main__":
    # Test Orchestrator (Mock run)
    pass
