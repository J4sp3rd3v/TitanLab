import asyncio
import edge_tts
import os

class TitanVoice:
    def __init__(self):
        self.output_dir = "audio_assets"
        os.makedirs(self.output_dir, exist_ok=True)
        # Voices optimized for viral content
        self.voices = {
            "Male Deep (Horror)": "en-US-ChristopherNeural",
            "Female Soft (Story)": "en-US-AriaNeural",
            "Male Energetic (Viral)": "en-US-GuyNeural",
            "Female Energetic (Viral)": "en-US-JennyNeural"
        }

    async def generate_speech(self, text, voice_key="Male Deep (Horror)", filename="output.mp3"):
        """
        Generates speech from text using edge-tts.
        """
        voice = self.voices.get(voice_key, "en-US-ChristopherNeural")
        output_path = os.path.join(self.output_dir, filename)
        
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        
        print(f"🎙️ AUDIO GENERATED: {output_path}")
        return output_path

    def get_voices(self):
        return list(self.voices.keys())

# Helper for synchronous calls (Colab compatibility)
def run_tts(text, voice_key, filename="output.mp3"):
    engine = TitanVoice()
    return asyncio.run(engine.generate_speech(text, voice_key, filename))
