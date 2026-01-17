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

    def generate_full_movie_auto(self, prompt, duration_min=1):
        """
        Generates a short movie based on a single prompt.
        Automates: Script -> Scenes -> Voice -> Video -> Editing.
        """
        print(f"🎬 STARTING AUTO-MOVIE PRODUCTION: '{prompt}'")
        
        # 1. Simple Story Logic (In a real scenario, use LLM here)
        # We simulate a 3-act structure based on the prompt
        scenes = [
            {"desc": f"Establishing shot, {prompt}, cinematic, wide angle", "narration": f"In a world where {prompt} exists..."},
            {"desc": f"Close up of main character, {prompt}, emotional, detailed", "narration": "One journey begins..."},
            {"desc": f"Action sequence, {prompt}, dynamic motion, fast", "narration": "And nothing will ever be the same."}
        ]
        
        video_clips = []
        audio_clips = []
        
        # 2. Production Loop
        for i, scene in enumerate(scenes):
            print(f"   [Act {i+1}] Action!")
            
            # Audio
            audio_path = f"auto_voice_{i}.mp3"
            run_tts(scene["narration"], "Male Deep (Horror)", audio_path)
            audio_clips.append(os.path.abspath(f"audio_assets/{audio_path}"))
            
            # Video (Using God Mode for best quality)
            # Create a consistent character first if needed, but for 'Movie' we often want general shots
            # We use God Mode (CogVideoX) for high coherence
            vid = self.director.generate_god_mode(scene["desc"])
            video_clips.append(os.path.abspath(vid))
            
        # 3. Final Edit
        # Create a simple montage
        # (For now, just return the list of clips, as TitanEditor stitching needs update)
        print("✅ PRODUCTION WRAP. Returning assets.")
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
