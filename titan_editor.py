from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
import os
import random

class TitanEditor:
    def __init__(self):
        self.output_dir = "final_renders"
        os.makedirs(self.output_dir, exist_ok=True)

    def create_viral_short(self, video_paths, audio_path, output_filename="viral_short.mp4"):
        """
        Combines multiple video clips and syncs them with a voiceover.
        Loops video clips if audio is longer.
        """
        try:
            print("🎬 STARTING VIDEO EDITING...")
            
            # Load Audio
            audio = AudioFileClip(audio_path)
            audio_duration = audio.duration
            print(f"   -> Audio Duration: {audio_duration:.2f}s")

            # Load Videos
            clips = []
            current_duration = 0
            
            # Simple logic: cycle through videos until we match audio duration
            while current_duration < audio_duration:
                for v_path in video_paths:
                    if current_duration >= audio_duration:
                        break
                    
                    clip = VideoFileClip(v_path)
                    # Resize for 9:16 (Shorts/Reels) if needed, usually generated as such
                    # clip = clip.resize(height=1920) 
                    # clip = clip.crop(x1=..., y1=..., width=1080, height=1920)
                    
                    clips.append(clip)
                    current_duration += clip.duration
            
            # Concatenate
            final_video = concatenate_videoclips(clips, method="compose")
            
            # Trim to audio length
            final_video = final_video.subclip(0, audio_duration)
            
            # Add Audio
            final_video = final_video.set_audio(audio)
            
            # Export
            output_path = os.path.join(self.output_dir, output_filename)
            final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")
            
            print(f"✨ VIRAL VIDEO RENDERED: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"<!> EDITING ERROR: {e}")
            return None

    def add_background_music(self, video_path, music_path, volume=0.1):
        # Implementation for adding bg music
        pass
