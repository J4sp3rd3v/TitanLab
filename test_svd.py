import os
from titan_video_engine import TitanDirector

def test_svd():
    director = TitanDirector()
    
    # 1. Create a dummy character
    print("Creating dummy character...")
    char_path = director.create_character("A cinematic photo of a cyberpunk girl, neon lights, 8k")
    
    # 2. Run SVD
    print("Running SVD-XT Realism Test...")
    try:
        video = director.generate_realism(fps=7)
        if video and os.path.exists(video):
            print(f"SUCCESS: {video}")
        else:
            print("FAILURE: No video generated")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_svd()
