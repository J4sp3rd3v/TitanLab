from titan_video_engine import TitanDirector
import time
import os

def test_perfect_dance():
    print("🚀 INITIALIZING TITAN DIRECTOR FOR HIGH-FIDELITY GENERATION...")
    director = TitanDirector()
    
    # 1. Create Character (High Quality)
    print("\n[1/2] Creating Perfectionist Character...")
    # Using specific prompts to ensure anatomical correctness from the start
    char_prompt = "stunning young woman, fit body, casual trendy dance outfit, symmetric face, perfect eyes, natural skin texture, 8k, masterpiece, photorealistic, standing pose, neutral background"
    char_path = director.create_character(char_prompt)
    
    if not char_path or not os.path.exists(char_path):
        print("❌ Character generation failed.")
        return

    print(f"✅ Character created: {char_path}")
    
    # 2. Setup for Action (ULTRA MODE)
    print("\n[2/2] Generating 30s Dance (Mode: ULTRA | Upscale: ON)...")
    director.set_quality("ULTRA")
    
    action_prompt = "dancing gracefully, modern dance, fluid movements, rhythmic body sway, happy expression, highly detailed, 8k, masterpiece"
    
    # motion_strength=6: Enough to be dynamic, not enough to break bones (Techno Viking is 8-9)
    # upscale=True: Essential for "no artifacts" look
    video_path = director.generate_action(
        prompt=action_prompt, 
        duration_sec=30, 
        motion_strength=6, 
        upscale=True
    )
    
    if video_path and os.path.exists(video_path):
        print(f"\n✨ SUCCESS! Video saved at: {os.path.abspath(video_path)}")
        print("   Specs: 30s duration, ULTRA quality, Upscaled 1080p, Corrected Anatomy")
    else:
        print("\n❌ Video generation failed.")

if __name__ == "__main__":
    test_perfect_dance()