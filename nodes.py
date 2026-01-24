# ------------------------------------------------------------------------------
# VIBE: Visual Instruction Based Editor implementation for ComfyUI
# Original VIBE Model: AI-Forever / Alibaba / NVLabs
# ComfyUI Node Implementation: ato-zen
# Repository: https://github.com/ato-zen/ComfyUI-VIBE
# ------------------------------------------------------------------------------

import torch
import numpy as np
from PIL import Image
import os
import sys
import folder_paths
import threading
from server import PromptServer
from aiohttp import web

# 1. PATH CONFIGURATION
NODE_ROOT = os.path.dirname(os.path.abspath(__file__))
VIBE_MODELS_DIR = os.path.join(folder_paths.models_dir, "vibe")

# Ensure base directory exists on load
if not os.path.exists(VIBE_MODELS_DIR):
    os.makedirs(VIBE_MODELS_DIR, exist_ok=True)

if "vibe" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("vibe", VIBE_MODELS_DIR)

if NODE_ROOT not in sys.path:
    sys.path.insert(0, NODE_ROOT)

# 2. SERVER API FOR SMART DOWNLOAD / UPDATE
@PromptServer.instance.routes.post("/vibe/download_model")
async def download_vibe_api(request):
    try:
        # Check if 'huggingface_hub' is available
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            return web.json_response({
                "status": "error", 
                "message": "Module 'huggingface_hub' missing! Reinstall requirements."
            }, status=500)

        target_path = os.path.join(VIBE_MODELS_DIR, "VIBE-Image-Edit")
        
        # Decide initial state based on folder existence
        is_update = os.path.exists(target_path)
        action_type = "updating" if is_update else "downloading"

        def run_download():
            repo_id = "iitolstykh/VIBE-Image-Edit"
            print(f"🚀 VIBE: Starting {action_type} process for {target_path}...")
            
            # Notify JS that we started actual heavy work
            PromptServer.instance.send_sync("vibe_status", {
                "status": "progress", 
                "message": f"{action_type.capitalize()}..."
            })

            try:
                # snapshot_download handles both download and update intelligently
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=target_path,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )

                print("✅ VIBE: Model ready.")
                PromptServer.instance.send_sync("vibe_status", {
                    "status": "finished", 
                    "message": "Model is ready and up-to-date."
                })

            except Exception as e:
                print(f"❌ VIBE: Operation failed: {e}")
                PromptServer.instance.send_sync("vibe_status", {
                    "status": "error", 
                    "message": f"Failed: {str(e)}"
                })

        threading.Thread(target=run_download, daemon=True).start()
        
        # Return immediate response so UI knows request was received
        return web.json_response({"status": "started", "mode": action_type})
        
    except Exception as e:
        return web.json_response({"status": "error", "message": f"Error: {str(e)}"}, status=500)

# 3. LAZY LOADING PLACEHOLDERS
ImageEditor = None
VIBE_MODEL_INSTANCE = None
VIBE_CURRENT_PATH = None

class VibeEditorNode:
    """
    ComfyUI Node for instruction-based image editing (VIBE).
    Supports automated resolution mapping from Latent or Image inputs.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("STRING", {"multiline": True, "default": "make it blue"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg_text": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "cfg_image": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "latent_image": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "VIBE"

    def _auto_find_model(self):
        """Finds the first valid VIBE directory in models/vibe/"""
        if not os.path.exists(VIBE_MODELS_DIR):
            return None
        for entry in os.listdir(VIBE_MODELS_DIR):
            full_path = os.path.join(VIBE_MODELS_DIR, entry)
            # Validation: Check for model_index.json
            if os.path.isdir(full_path) and "model_index.json" in os.listdir(full_path):
                return full_path
        return None

    def generate(self, positive, steps, cfg_text, cfg_image, seed, image=None, latent_image=None):
        global VIBE_MODEL_INSTANCE, VIBE_CURRENT_PATH, ImageEditor

        # Lazy Import to speed up initial ComfyUI startup
        if ImageEditor is None:
            try:
                print("⏳ VIBE: Importing core libraries...")
                from vibe.editor import ImageEditor
                print("✅ VIBE: Libraries imported.")
            except ImportError as e:
                print(f"❌ VIBE Load Error: {e}")
                raise ImportError(f"VIBE library missing! Details: {e}")

        # Resolve model path
        checkpoint_path = self._auto_find_model() or "iitolstykh/VIBE-Image-Edit"

        # Load or switch model
        if VIBE_MODEL_INSTANCE is None or VIBE_CURRENT_PATH != checkpoint_path:
            print(f"⏳ VIBE: Loading model from {checkpoint_path}")
            try:
                VIBE_MODEL_INSTANCE = ImageEditor(checkpoint_path=checkpoint_path, device="cuda")
                VIBE_CURRENT_PATH = checkpoint_path
                print("✅ VIBE: Model loaded.")
            except Exception as e:
                print(f"❌ VIBE: Failed to load model! {e}")
                raise RuntimeError(f"VIBE Model Load Error: {e}")

        # Resolution Logic (Latent > Image > Default)
        target_w, target_h = 1024, 1024
        if latent_image is not None:
            target_h = latent_image["samples"].shape[2] * 8
            target_w = latent_image["samples"].shape[3] * 8
        elif image is not None:
            target_h, target_w = image.shape[1], image.shape[2]

        # Enforce Sana-1.5 requirements (divisible by 32)
        target_w = (target_w // 32) * 32
        target_h = (target_h // 32) * 32

        # Prepare input image
        input_pil = None
        if image is not None:
            batch_data = 255. * image[0].cpu().numpy()
            input_pil = Image.fromarray(np.clip(batch_data, 0, 255).astype(np.uint8)).convert("RGB")
            
            # Resize input to match target resolution
            if input_pil.size != (target_w, target_h):
                input_pil = input_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Set deterministic state
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"🎨 VIBE: Generating... '{positive}' [{target_w}x{target_h}]")

        # Core Inference
        result = VIBE_MODEL_INSTANCE.generate_edited_image(
            instruction=positive,
            conditioning_image=input_pil,
            num_inference_steps=steps,
            guidance_scale=cfg_text,
            image_guidance_scale=cfg_image,
            seed=seed,
            randomize_seed=False,
            t2i_width=target_w,
            t2i_height=target_h,
            num_images_per_prompt=1
        )

        # Output Processing
        final_pil = result if isinstance(result, Image.Image) else result[0]
        final_pil = final_pil.convert("RGB")
        
        # Optional: Restore original aspect ratio if input image was provided
        if image is not None:
             orig_w, orig_h = image.shape[2], image.shape[1]
             if final_pil.size != (orig_w, orig_h):
                 final_pil = final_pil.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

        # Convert to ComfyUI Tensor [Batch, H, W, C]
        out_tensor = np.array(final_pil).astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(out_tensor)[None, ]

        return (out_tensor,)

# ------------------------------------------------------------------------------
# Auto-setup: Copy example image to ComfyUI input folder
# ------------------------------------------------------------------------------
try:
    import shutil
    EXAMPLE_FILE = "vibe_example_woman.png"
    my_examples_dir = os.path.join(NODE_ROOT, "examples")
    comfy_input_dir = folder_paths.get_input_directory()
    source_path = os.path.join(my_examples_dir, EXAMPLE_FILE)
    dest_path = os.path.join(comfy_input_dir, EXAMPLE_FILE)

    if os.path.exists(source_path) and not os.path.exists(dest_path):
        shutil.copy(source_path, dest_path)
except Exception:
    pass

# Node Registration
NODE_CLASS_MAPPINGS = { "VIBE_Editor": VibeEditorNode }
NODE_DISPLAY_NAME_MAPPINGS = { "VIBE_Editor": "VIBE Image Editor" }
WEB_DIRECTORY = "./js"
