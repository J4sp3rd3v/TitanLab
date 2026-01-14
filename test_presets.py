import os
from titan_video_engine import TitanDirector

def test_preset_generation():
    print("🧪 TESTING VIRAL PRESETS...")
    director = TitanDirector()
    
    # Test 1: List Presets
    presets = director.get_preset_list()
    print(f"📋 Available Presets: {presets}")
    if "✨ TikTok Perfect (Standard)" not in presets:
        print("❌ Preset list missing expected keys!")
        return
    
    # Test 2: Apply Preset Logic
    prompt = "girl"
    preset = "✨ TikTok Perfect (Standard)"
    full_prompt = director.apply_preset(prompt, preset)
    print(f"📝 Applied Preset '{preset}':\n   -> {full_prompt}")
    
    expected_fragment = "trending on tiktok"
    if expected_fragment not in full_prompt:
        print("❌ Preset application failed!")
        return

    # Test 3: Generate Character with Preset (Dry Run - just checking path creation)
    # Actually generating takes time. Let's just generate one to verify it runs.
    print("\n📸 Generating Character with 'Unique Beauty (Vitiligo)' preset...")
    
    preset_vitiligo = "🦋 Unique Beauty (Vitiligo)"
    final_prompt = director.apply_preset("close up portrait", preset_vitiligo)
    
    # We will simulate the generation call or just call it if we want to be sure.
    # Given the user wants "modelle perfette", running one generation is good validation.
    # It takes about 20-30s on GPU, or slower on CPU.
    # I'll rely on the previous success of create_character.
    
    path, filename = director.create_character(final_prompt)
    
    if os.path.exists(path):
        print(f"✅ Character Generated Successfully: {path}")
    else:
        print("❌ Character Generation Failed: File not found.")

if __name__ == "__main__":
    test_preset_generation()
