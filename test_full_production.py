import os
import time
from titan_video_engine import TitanDirector

def run_full_test():
    print("======================================================================")
    print("🚀 TITAN AI DIRECTOR - AUTOMATED SYSTEM TEST (END-TO-END)")
    print("======================================================================")
    
    director = TitanDirector()
    
    # ------------------------------------------------------------------------
    # STEP 1: CHARACTER GENERATION (CYBERREALISTIC PONY)
    # ------------------------------------------------------------------------
    print("\n[STEP 1/3] 👤 GENERATING HIGH-FIDELITY CHARACTER...")
    char_prompt = "stunning woman, fit body, casual trendy streetwear, symmetric face, perfect eyes, natural skin texture, standing pose, neutral background, looking at camera"
    
    try:
        char_path = director.create_character(char_prompt)
        if not char_path or not os.path.exists(char_path):
            print("❌ FAILURE: Character generation failed.")
            return
        print(f"✅ SUCCESS: Character saved at {char_path}")
    except Exception as e:
        print(f"❌ ERROR in Step 1: {e}")
        return

    # ------------------------------------------------------------------------
    # STEP 2: CONFIGURE SETTINGS
    # ------------------------------------------------------------------------
    print("\n[STEP 2/3] ⚙️ CONFIGURING ULTRA QUALITY MODE...")
    director.set_quality("ULTRA")
    
    # ------------------------------------------------------------------------
    # STEP 3: ACTION GENERATION (FULL PRODUCTION)
    # ------------------------------------------------------------------------
    print("\n[STEP 3/3] 🎬 GENERATING 30s ACTION VIDEO (FULL PRODUCTION)...")
    action_prompt = "dancing energetically, shuffle dance, fluid body movement, rhythmic, 8k, masterpiece"
    
    # FULL DURATION
    duration = 30 
    
    try:
        # Use CINEMA mode (Best Balance of Quality/Speed) + Upscale
        director.set_quality("CINEMA")
        
        video_path = director.generate_action(
            prompt=action_prompt,
            duration_sec=duration,
            motion_strength=6, 
            upscale=True,
            progress_callback=lambda i, n, msg: print(f"   [Progress {i}/{n}] {msg}")
        )
        
        if video_path and os.path.exists(video_path):
            print(f"\n✨ COMPLETE SUCCESS! Video generated: {video_path}")
            print(f"   Specs: {duration}s Duration | CINEMA Mode | UPSCALED")
        else:
            print("\n❌ FAILURE: Video file not found.")
            
    except Exception as e:
        print(f"❌ ERROR in Step 3: {e}")

if __name__ == "__main__":
    run_full_test()
