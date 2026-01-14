import os
from titan_video_engine import TitanDirector

def run_test():
    print("🍌 TITAN TEST: BANANA DANCE PROTOCOL INITIATED 🍌")
    
    # 1. Initialize Director
    director = TitanDirector()
    
    # 2. Create Character (Identity)
    print("\n[1/3] Creating Character Identity...")
    char_prompt = "A cute girl with long hair, wearing casual clothes, realistic, 8k, photorealistic, full body shot"
    char_path = director.create_character(char_prompt)
    print(f"✅ Character Created: {char_path}")
    
    # 3. Generate Action (Higgsfield Style)
    print("\n[2/3] Generating Action Video (30 Seconds)...")
    # Using 'CINEMA' mode for best motion quality (FreeInit)
    director.set_quality("CINEMA") 
    
    action_prompt = "energetic dancing on a giant banana, dynamic motion, spinning, jumping, legs moving, funny, surreal, highly detailed, 4k"
    video_path = director.generate_action(action_prompt, duration_sec=30)
    
    if video_path and os.path.exists(video_path):
        print(f"\n✅ SUCCESS! Video generated at: {video_path}")
    else:
        print("\n❌ ERROR: Video generation failed.")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"\n❌ CRITICAL FAILURE: {str(e)}")
        import traceback
        traceback.print_exc()
