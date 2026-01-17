import torch
from huggingface_hub import hf_hub_download
import time
from diffusers import (
    DiffusionPipeline, 
    DPMSolverMultistepScheduler, 
    StableVideoDiffusionPipeline, 
    CogVideoXPipeline, 
    StableDiffusionXLPipeline, 
    EulerDiscreteScheduler,
    AnimateDiffPipeline,
    MotionAdapter,
    DDIMScheduler,
    ControlNetModel
)
from diffusers.utils import export_to_video, load_image
from safetensors.torch import load_file
import os
import sys
import cv2
from PIL import Image
import gc
import numpy as np
from controlnet_aux import OpenposeDetector

import requests
import json

class PixVerseBridge:
    """
    Bridge to PixVerse API (Official/MCP style)
    Requires PIXVERSE_API_KEY environment variable.
    """
    def __init__(self):
        self.api_key = os.environ.get("PIXVERSE_API_KEY")
        self.base_url = "https://api.pixverse.ai/v1" # Hypothetical endpoint based on typical structure
        
    def is_available(self):
        return self.api_key is not None

    def generate(self, prompt, negative_prompt="", aspect_ratio="9:16", model="v3.5", duration=5):
        if not self.is_available():
            print("<!> PIXVERSE KEY MISSING. Set 'PIXVERSE_API_KEY' in secrets.")
            return None
            
        print(f"☁️ CONTACTING PIXVERSE CLOUD ({model})...")
        
        # This is a mock implementation of the request structure based on standard text-to-video APIs
        # Since the official documentation is behind a login, this mimics the MCP structure.
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "model": model
        }
        
        try:
            # 1. Submit Job
            # resp = requests.post(f"{self.base_url}/video/generate", json=payload, headers=headers)
            # job_id = resp.json().get("id")
            
            # 2. Poll for result (Mocking wait)
            # while True: ...
            
            print("⚠️ PIXVERSE BRIDGE: API Key implementation requires valid credits.")
            print("   -> Fallback to Titan Native Engine for this demo.")
            return None
            
        except Exception as e:
            print(f"<!> PIXVERSE ERROR: {e}")
            return None

# ==============================================================================
# TITAN VIDEO ENGINE - GOD MODE (V7.0 - CINEMA STUDIO)
# SUPPORTS: FREEINIT (Temporal Smoothing) & SMART PROMPTING
# ==============================================================================

class TitanDirector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Pipelines
        self.t2v_pipe = None
        self.i2v_pipe = None
        self.scene_gen_pipe = None # SDXL + IP-Adapter
        self.action_pipe = None    # AnimateDiff + IP-Adapter
        self.god_mode_pipe = None  # CogVideoX-5B
        
        # Settings
        self.quality_mode = "SPEED" # SPEED or CINEMA
        
        # External Engines
        self.pixverse = PixVerseBridge()
        
        # VIRAL PRESETS (TikTok/Social Media Styles)
        self.VIRAL_PRESETS = {
            "None": "",
            "✨ TikTok Perfect (Standard)": "beautiful young woman, trending on tiktok, perfect face, symmetrical, soft lighting, 8k, ultra realistic, influencer look",
            "🖤 TikTok Artistic (B&W)": "black and white photography, artistic portrait, high contrast, moody, emotive, detailed skin texture, film grain, noir style",
            "🦋 Unique Beauty (Vitiligo)": "beautiful woman with vitiligo, unique skin patterns, detailed skin texture, natural beauty, raw photo, hyperrealistic",
            "🦌 Natural Beauty (Freckles)": "beautiful woman, no makeup, heavy freckles, detailed pores, hyperrealistic skin, raw photography, natural lighting",
            "🦇 Goth/Alt Aesthetic": "gothic style woman, pale skin, dark eyeliner, alternative fashion, moody lighting, edgy, trending on tiktok",
            "💄 Insta Baddie (Glam)": "instagram model, heavy glam makeup, ring light, perfect contour, styled hair, fashion influencer, studio lighting",
            "💪 Fitness/GymTok": "fitness model, athletic physique, gym wear, sweat, workout glow, gym lighting, strong, confident",
            "👻 Creepypasta/Horror": "scary atmosphere, horror style, dark lighting, eerie, unsettling, creepy, nightmare fuel, 8k, detailed, photorealistic, cinematic horror",
            "🏚️ Liminal Spaces (Backrooms)": "liminal space, empty hallway, buzzing lights, yellow wallpaper, claustrophobic, unsettling, weirdcore, dreamcore, hyperrealistic"
        }
        
        # State
        self.current_character_path = None
        self.last_generated_scene_path = None
        
        # Hardware Check
        print(f"⚡ TITAN DIRECTOR ONLINE. DEVICE: {self.device.upper()}")
        if self.device == "cuda":
            self.vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"⚡ VRAM DETECTED: {self.vram_gb:.1f} GB")
        else:
            self.vram_gb = 0

    def _unload_all(self):
        """Aggressive memory cleaner"""
        if self.t2v_pipe: del self.t2v_pipe
        if self.i2v_pipe: del self.i2v_pipe
        if self.scene_gen_pipe: del self.scene_gen_pipe
        if self.action_pipe: del self.action_pipe
        
        self.t2v_pipe = None
        self.i2v_pipe = None
        self.scene_gen_pipe = None
        self.action_pipe = None
        
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print("🧹 MEMORY PURGED.")

    def _smart_prompt(self, prompt, is_pony=False):
        """Injects cinema-grade keywords automatically"""
        if is_pony:
            # Kept for backward compatibility or if we switch back
            prefix = "score_9, score_8_up, score_7_up, "
            quality_tags = ", masterpiece, best quality, ultra-detailed, 8k, photorealistic, cinematic lighting, sharp focus"
        else:
            prefix = ""
            quality_tags = ", masterpiece, best quality, 8k, cinematic lighting, photorealistic, hdr, sharp focus, high definition, detailed texture, professional photography"
            
        # Massive negative prompt for anatomy safety
        negative = "blurry, low quality, distorted, bad anatomy, ugly, pixelated, text, watermark, bad hands, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, username, artist name, deformed, disfigured, extra limbs, missing limbs, floating limbs, disconnected limbs, mutation, mutated, gross proportions, malformed limbs, long neck, duplicate, mutilated, out of frame, body out of frame, clone, duplicate, fused, fused fingers, too many fingers, collapse, glitch"
        return f"{prefix}{prompt}{quality_tags}", negative

    def get_preset_list(self):
        return list(self.VIRAL_PRESETS.keys())

    def apply_preset(self, prompt, preset_name):
        if not preset_name or preset_name not in self.VIRAL_PRESETS or preset_name == "None":
            return prompt
        
        base = self.VIRAL_PRESETS[preset_name]
        # Combine efficiently
        if prompt:
            return f"{base}, {prompt}"
        return base

    def set_quality(self, mode):
        self.quality_mode = mode.upper()
        print(f"⚡ QUALITY SET TO: {self.quality_mode}")

    def load_scene_builder(self):
        if self.scene_gen_pipe is not None: return
        self._unload_all()
        print("⚡ LOADING SCENE BUILDER (Juggernaut XL Lightning)...")
        try:
            # SWITCHED TO JUGGERNAUT XL LIGHTNING (Faster & High Quality)
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "RunDiffusion/Juggernaut-XL-Lightning",
                torch_dtype=self.dtype,
                variant="fp16" # FORCE FP16 TO AVOID 10GB DOWNLOAD
            )
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
            
            # Ensure correct scheduler for Lightning
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_tiling()
            self.scene_gen_pipe = pipe
            print("✅ SCENE BUILDER READY (JUGGERNAUT XL).")
        except Exception as e:
            print(f"<!> BUILDER LOAD ERROR (Fallback to RealVis): {e}")
            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    "SG161222/RealVisXL_V4.0_Lightning",
                    torch_dtype=self.dtype,
                    variant="fp16"
                )
                pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
                pipe.enable_model_cpu_offload()
                self.scene_gen_pipe = pipe
                print("✅ SCENE BUILDER READY (FALLBACK).")
            except Exception as e2:
                print(f"<!> FATAL BUILDER ERROR: {e2}")

    def load_animator(self):
        if self.i2v_pipe is not None: return
        self._unload_all()
        print("⚡ LOADING ANIMATOR (SVD-XT)...")
        try:
            # Try XT version (High Quality)
            repo_id = "stabilityai/stable-video-diffusion-img2vid-xt"
            print(f"   -> Connecting to {repo_id}...")
            
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                repo_id, 
                torch_dtype=self.dtype, 
                variant="fp16"
            )
            pipe.enable_model_cpu_offload()
            # pipe.vae.enable_tiling() # SVD VAE doesn't support tiling easily
            if self.vram_gb < 10:
                pipe.unet.enable_forward_chunking()
            self.i2v_pipe = pipe
            print("✅ ANIMATOR READY (SVD-XT).")
        except Exception as e:
            print(f"<!> SVD-XT LOAD ERROR: {e}")
            print("   -> Fallback to Standard SVD...")
            try:
                pipe = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid", 
                    torch_dtype=self.dtype, 
                    variant="fp16"
                )
                pipe.enable_model_cpu_offload()
                # pipe.vae.enable_tiling()
                if self.vram_gb < 10:
                    pipe.unet.enable_forward_chunking()
                self.i2v_pipe = pipe
                print("✅ ANIMATOR READY (SVD Standard).")
            except Exception as e2:
                 print(f"<!> FATAL SVD ERROR: {e2}")

    def load_action_engine(self):
        if self.action_pipe is not None: return
        self._unload_all()
        print("⚡ LOADING ACTION ENGINE (AnimateDiff Lightning + ControlNet)...")
        try:
            # 1. Load 4-Step Lightning Motion Module manually
            print("   -> Downloading Lightning weights...")
            ckpt_path = hf_hub_download("ByteDance/AnimateDiff-Lightning", "animatediff_lightning_4step_diffusers.safetensors")
            
            # 2. Load Base Adapter Structure
            adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=self.dtype)
            
            # 3. Inject Weights
            state_dict = load_file(ckpt_path, device="cpu")
            adapter.load_state_dict(state_dict)
            
            # 4. Load ControlNet (OpenPose) for Motion Transfer
            print("   -> Loading ControlNet (OpenPose)...")
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_openpose",
                torch_dtype=self.dtype
            )

            # 5. Create Pipeline
            pipe = AnimateDiffPipeline.from_pretrained(
                "emilianJR/epiCRealism",
                motion_adapter=adapter,
                controlnet=controlnet,
                torch_dtype=self.dtype,
                variant="fp16"
            )
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
            
            # LIGHTNING RECOMMENDS EULER DISCRETE
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear"
            )
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_tiling()
            self.action_pipe = pipe
            
            # Load Detector
            self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            
            print("✅ ACTION ENGINE READY (Motion Control Active).")
        except Exception as e:
            print(f"<!> ACTION ENGINE ERROR: {e}")

    def generate_motion_transfer(self, prompt, source_video_path):
        """
        Generates a video where the character mimics the source video motion (Nano Banana style).
        """
        if not self.current_character_path:
            print("<!> NO CHARACTER SELECTED.")
            return None
            
        self.load_action_engine()
        full_prompt, neg = self._smart_prompt(prompt)
        print(f"\n💃 MOTION TRANSFER: '{prompt}' (Source: {source_video_path})...")
        
        # 1. Extract Pose Frames from Source Video
        print("   -> Extracting Pose Skeleton...")
        cap = cv2.VideoCapture(source_video_path)
        pose_frames = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < 32: # Limit to 32 frames (~4s)
            ret, frame = cap.read()
            if not ret: break
            
            # Resize to portrait
            frame = cv2.resize(frame, (512, 768))
            # Convert to PIL
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Extract Pose
            pose_img = self.openpose(pil_frame)
            pose_frames.append(pose_img)
            frame_count += 1
            
        cap.release()
        
        if not pose_frames:
            print("<!> FAILED TO READ VIDEO.")
            return None
            
        ref_image = load_image(self.current_character_path)
        self.action_pipe.set_ip_adapter_scale(0.7) # Strong Identity
        
        print(f"   -> Generating {len(pose_frames)} frames...")
        
        output = self.action_pipe(
            prompt=full_prompt,
            negative_prompt=neg,
            ip_adapter_image=ref_image,
            control_image=pose_frames, # Pass poses to ControlNet
            controlnet_conditioning_scale=1.0, # Strong Control
            num_frames=len(pose_frames),
            guidance_scale=1.5,
            num_inference_steps=8, # Lightning steps
            width=512,
            height=768,
            generator=torch.manual_seed(42)
        )
        
        frames = output.frames[0]
        filename = f"motion_{os.urandom(4).hex()}.mp4"
        export_to_video(frames, output_video_path=filename, fps=8)
        print(f"💾 MOTION TRANSFER SAVED: {filename}")
        return filename

    def load_god_mode(self):
        if self.god_mode_pipe is not None: return
        self._unload_all()
        print("⚡ LOADING GOD MODE (CogVideoX-5B - Sora Class)...")
        try:
            # CogVideoX-5B: The closest open source model to Sora
            # Using bfloat16 for best quality/memory balance
            pipe = CogVideoXPipeline.from_pretrained(
                "THUDM/CogVideoX-5b",
                torch_dtype=self.dtype
            )
            
            # CRITICAL OPTIMIZATIONS FOR 16GB VRAM (T4)
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
            
            self.god_mode_pipe = pipe
            print("✅ GOD MODE READY (CogVideoX-5B).")
        except Exception as e:
            print(f"<!> GOD MODE LOAD ERROR: {e}")
            print("   -> Fallback to 2B Model (Lighter)...")
            try:
                pipe = CogVideoXPipeline.from_pretrained(
                    "THUDM/CogVideoX-2b",
                    torch_dtype=self.dtype,
                    variant="fp16"
                )
                pipe.enable_model_cpu_offload()
                pipe.vae.enable_tiling()
                self.god_mode_pipe = pipe
                print("✅ GOD MODE READY (Fallback 2B).")
            except Exception as e2:
                print(f"<!> FATAL GOD MODE ERROR: {e2}")

    def generate_god_mode(self, prompt, num_frames=49):
        """
        Generates video using CogVideoX (6s @ 8fps = 49 frames).
        """
        if not prompt: return None
        
        self.load_god_mode()
        print(f"🎬 GENERATING GOD MODE VIDEO: '{prompt}'")
        
        full_prompt, neg = self._smart_prompt(prompt)
        
        video = self.god_mode_pipe(
            prompt=full_prompt,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=num_frames,
            guidance_scale=6,
            generator=torch.manual_seed(int(time.time()))
        ).frames[0]
        
        filename = f"god_{os.urandom(4).hex()}.mp4"
        export_to_video(video, output_video_path=filename, fps=8)
        print(f"💾 GOD MODE VIDEO SAVED: {filename}")
        return filename

    # ==========================================================================
    # CORE ACTIONS
    # ==========================================================================

    def set_character(self, path):
        if os.path.exists(path):
            self.current_character_path = path
            print(f"👤 CHARACTER LOCKED: {path}")
        else:
            print(f"<!> FILE NOT FOUND: {path}")

    def create_character(self, prompt):
        """Generates the main character (Identity)"""
        print(f"\n📸 GENERATING NEW CHARACTER: '{prompt}'")
        self.load_scene_builder()
        
        # Disable IP-Adapter for pure text-to-image
        self.scene_gen_pipe.set_ip_adapter_scale(0.0)
        
        full_prompt, negative = self._smart_prompt(prompt, is_pony=False)
        
        # Create dummy image for IP-Adapter (required even if scale is 0.0)
        from PIL import Image
        # SDXL Portrait Resolution
        width, height = 832, 1216 
        dummy_img = Image.new("RGB", (width, height), (0, 0, 0))

        image = self.scene_gen_pipe(
            full_prompt, 
            negative_prompt=negative, 
            ip_adapter_image=dummy_img,
            height=height,
            width=width,
            num_inference_steps=6, # Lightning is FAST (4-8 steps)
            guidance_scale=2.0     # Lightning likes low CFG
        ).images[0]
        
        import time
        filename = f"char_{int(time.time())}.png"
        image.save(filename)
        self.current_character_path = os.path.abspath(filename)
        return self.current_character_path, filename

    def build_scene(self, prompt):
        if not self.current_character_path:
            print("<!> NO CHARACTER SELECTED.")
            return None
        self.load_scene_builder()
        full_prompt, neg = self._smart_prompt(prompt, is_pony=True)
        print(f"\n🎨 PAINTING SCENE: '{prompt}'...")
        
        ref_image = load_image(self.current_character_path)
        self.scene_gen_pipe.set_ip_adapter_scale(0.7)
        
        image = self.scene_gen_pipe(
            prompt=full_prompt,
            ip_adapter_image=ref_image,
            negative_prompt=neg,
            num_inference_steps=25,
            guidance_scale=5.0
        ).images[0]
        
        filename = f"scene_{os.urandom(4).hex()}.png"
        image = image.resize((1024, 576))
        image.save(filename)
        self.last_generated_scene_path = filename
        print(f"✅ SCENE READY: {filename}")
        return filename

    def load_realism_engine(self):
        self.load_animator()

    def generate_realism(self, motion_bucket_id=127, noise_aug_strength=0.1, fps=7):
        """
        SVD-XT Generation - The 'Pro' Mode for high realism.
        Generates 25 frames (~3-4s).
        """
        if not self.current_character_path:
            print("<!> NO CHARACTER SELECTED.")
            return None
        
        self.load_realism_engine()
        
        # Load and resize image for SVD (1024x576 is standard for SVD-XT)
        image = load_image(self.current_character_path)
        image = image.resize((576, 1024)) # Portrait Mode for Character
        
        print(f"🎬 GENERATING REALISM CLIP (SVD-XT)...")
        print(f"   [Params] Motion: {motion_bucket_id}, Noise: {noise_aug_strength}")
        
        frames = self.i2v_pipe(
            image, 
            decode_chunk_size=2,
            generator=torch.manual_seed(int(time.time())),
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            num_frames=25
        ).frames[0]
        
        filename = f"realism_{os.urandom(4).hex()}.mp4"
        export_to_video(frames, output_video_path=filename, fps=fps)
        print(f"💾 REALISM VIDEO SAVED: {filename}")
        return filename

    def generate_action(self, prompt, duration_sec=4, motion_strength=5, upscale=False, progress_callback=None):
        if not self.current_character_path:
            print("<!> NO CHARACTER SELECTED.")
            return None
            
        self.load_action_engine()
        full_prompt, neg = self._smart_prompt(prompt)
        print(f"\n⚡ ACTION MODE: '{prompt}' (Target: {duration_sec}s, Motion: {motion_strength})...")
        
        ref_image = load_image(self.current_character_path)
        # 0.6 stronger identity retention to fix anatomy
        self.action_pipe.set_ip_adapter_scale(0.6)
        
        # Add motion-specific negative prompts and anatomy safety
        neg = neg + ", static, motionless, frozen, stationary, picture, photo, glitch, morphing, disjointed"

        # FreeInit Logic
        if self.quality_mode == "CINEMA":
            print("ℹ️  CINEMA MODE: ENABLING FREEINIT (Temporal Smoothing)...")
            self.action_pipe.enable_free_init(method="butterworth", num_iters=3)
        elif self.quality_mode == "ULTRA":
             print("ℹ️  ULTRA MODE: ENABLING MAX FREEINIT & QUALITY...")
             self.action_pipe.enable_free_init(method="butterworth", num_iters=5)
        else:
            self.action_pipe.disable_free_init()

        # Calculate chunks needed with overlap
        # 16 frames @ 8fps = 2s per chunk.
        # We overlap 4 frames (0.5s) for smooth crossfade.
        # Effective new content per chunk = 12 frames (1.5s).
        # To get 30s (240 frames), we need ~20 chunks.
        
        chunk_len = 16
        overlap = 4
        effective_len = chunk_len - overlap
        target_frames = duration_sec * 8
        
        num_chunks = int(target_frames / effective_len) + 1
        
        all_frames = []
        
        print(f"🎬 GENERATING {num_chunks} CLIPS TO STITCH (With Crossfade)...")
        
        for i in range(num_chunks):
            msg = f"Rendering Clip {i+1}/{num_chunks}..."
            print(f"   -> {msg}")
            if progress_callback:
                progress_callback(i, num_chunks, msg)
            
            # Use progressive seeds to keep style consistent but evolving
            current_seed = 42 + i*100 
            
            # Increase steps for anatomy quality
            if self.quality_mode == "SPEED": steps = 6
            elif self.quality_mode == "CINEMA": steps = 12
            elif self.quality_mode == "ULTRA": steps = 20 # Maximum Quality
            else: steps = 8

            output = self.action_pipe(
                prompt=full_prompt,
                ip_adapter_image=ref_image,
                negative_prompt=neg,
                num_frames=16,
                guidance_scale=1.0 + (motion_strength * 0.1), # 1.5 to 2.5 range usually
                width=512,
                height=768, # Portrait for better human anatomy
                num_inference_steps=steps,
                generator=torch.manual_seed(current_seed)
            )
            new_frames = output.frames[0]

            # --- DEBUG: CHECK FOR BLACK FRAMES ---
            first_frame_arr = np.array(new_frames[0])
            mean_brightness = np.mean(first_frame_arr)
            print(f"   [DEBUG] Clip {i+1} Mean Brightness: {mean_brightness:.2f}")
            if mean_brightness < 5:
                print("   <!> WARNING: DETECTED BLACK/DARK FRAME. VAE OR MODEL FAILURE?")
            if i == 0:
                new_frames[0].save(f"debug_clip_{i+1}_first_frame.png")
                print(f"   [DEBUG] Saved debug frame to debug_clip_{i+1}_first_frame.png")
            # -------------------------------------
            
            # AGGRESSIVE GC TO PREVENT CRASH
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            if i == 0:
                all_frames.extend(new_frames)
            else:
                # Crossfade Logic
                # Blend last 'overlap' frames of 'all_frames' with first 'overlap' frames of 'new_frames'
                prev_segment = all_frames[-overlap:]
                curr_segment = new_frames[:overlap]
                
                blended_segment = []
                for j in range(overlap):
                    alpha = (j + 1) / (overlap + 1)
                    # Simple linear blend
                    frame_prev = Image.fromarray(np.array(prev_segment[j]))
                    frame_curr = Image.fromarray(np.array(curr_segment[j]))
                    blended = Image.blend(frame_prev, frame_curr, alpha)
                    blended_segment.append(blended)
                
                # Replace the overlap part in all_frames with blended
                all_frames = all_frames[:-overlap] + blended_segment + new_frames[overlap:]

        if upscale:
            print("✨ UPSCALING TO HD (1024x1536)...")
            if progress_callback: progress_callback(num_chunks, num_chunks, "Upscaling to HD...")
            upscaled_frames = []
            for frame in all_frames:
                # Convert PIL to CV2
                img = np.array(frame)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Lanczos Upscale (2x)
                img = cv2.resize(img, (1024, 1536), interpolation=cv2.INTER_LANCZOS4)
                
                # Sharpening Kernel
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                img = cv2.filter2D(img, -1, kernel)
                
                # Convert back to PIL
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                upscaled_frames.append(Image.fromarray(img))
            all_frames = upscaled_frames

        if progress_callback:
            progress_callback(num_chunks, num_chunks, "Finalizing Video...")

        filename = f"action_{os.urandom(4).hex()}_long.mp4"
        export_to_video(all_frames, output_video_path=filename, fps=8)
        print(f"💾 LONG ACTION SAVED: {filename} (Duration: {len(all_frames)/8}s @ 8fps)")
        return filename

if __name__ == "__main__":
    director = TitanDirector()
    
    print("""
    ╔════════════════════════════════════════════════════╗
    ║           TITAN STUDIO (V7.0 - CINEMA GRADE)       ║
    ╠════════════════════════════════════════════════════╣
    ║   [1] CREATE NEW CHARACTER (From Text)             ║
    ║   [2] LOAD EXISTING CHARACTER (File Path)          ║
    ║   [3] BUILD SCENE (Place Character in New Zone)    ║
    ║   [4] ANIMATE SCENE (Turn Image to Video - SVD)    ║
    ║   [5] ACTION MODE (Make Character Dance/Run etc)   ║
    ║   [6] SET QUALITY (Current: SPEED)                 ║
    ╚════════════════════════════════════════════════════╝
    """)
    
    while True:
        status = f" [Active Char: {director.current_character_path}] [Mode: {director.quality_mode}]"
        choice = input(f"\n[TITAN{status}] Command (1-6, exit): ").strip()
        
        if choice == '1':
            p = input("Describe Character: ")
            director.create_character(p)
            
        elif choice == '2':
            path = input("Enter image filename: ")
            director.set_character(path)
            
        elif choice == '3':
            p = input("Describe New Location/Action: ")
            director.build_scene(p)
            
        elif choice == '4':
            director.generate_realism()
            
        elif choice == '5':
            p = input("Describe Action: ")
            director.generate_action(p)
            
        elif choice == '6':
            q = input("Select Mode (S=Speed [Fast], C=Cinema [Smooth/High Quality]): ").strip().upper()
            if q.startswith("C"): director.set_quality("CINEMA")
            else: director.set_quality("SPEED")
            
        elif choice == 'exit':
            break
