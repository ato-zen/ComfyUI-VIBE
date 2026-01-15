# ------------------------------------------------------------------------------
# VIBE: Visual Instruction Based Editor implementation for ComfyUI
# Implementation: ato-zen
# Repository: https://github.com/ato-zen/ComfyUI-VIBE
# ------------------------------------------------------------------------------

import torch
import numpy as np
from PIL import Image
import os
import sys
import folder_paths

# Registration of model directory in ComfyUI structure
VIBE_MODELS_DIR = os.path.join(folder_paths.models_dir, "vibe")
if not os.path.exists(VIBE_MODELS_DIR):
    os.makedirs(VIBE_MODELS_DIR, exist_ok=True)

# Add node directory to sys.path for internal module resolution
NODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if NODE_ROOT not in sys.path:
    sys.path.insert(0, NODE_ROOT)

try:
    from vibe.editor import ImageEditor
except ImportError as e:
    print(f"VIBE Load Error: Internal 'vibe' module not found. {e}")
    ImageEditor = None

VIBE_MODEL_INSTANCE = None
VIBE_CURRENT_PATH = None

class VibeEditorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("STRING", {"multiline": True, "default": "make it blue"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg_text": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "cfg_image": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 5.0, "step": 0.1}),
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
        """Finds the first folder in models/vibe containing model_index.json"""
        if not os.path.exists(VIBE_MODELS_DIR):
            return None
        for entry in os.listdir(VIBE_MODELS_DIR):
            full_path = os.path.join(VIBE_MODELS_DIR, entry)
            if os.path.isdir(full_path) and "model_index.json" in os.listdir(full_path):
                return full_path
        return None

    def generate(self, positive, steps, cfg_text, cfg_image, seed, image=None, latent_image=None):
        global VIBE_MODEL_INSTANCE, VIBE_CURRENT_PATH

        if ImageEditor is None:
            raise ImportError("VIBE library missing in custom_nodes folder.")

        checkpoint_path = self._auto_find_model() or "iitolstykh/VIBE-Image-Edit"

        if VIBE_MODEL_INSTANCE is None or VIBE_CURRENT_PATH != checkpoint_path:
            print(f"VIBE: Loading model from {checkpoint_path}")
            VIBE_MODEL_INSTANCE = ImageEditor(checkpoint_path=checkpoint_path, device="cuda")
            VIBE_CURRENT_PATH = checkpoint_path

        # RESOLUTION LOGIC: Priority is Latent > Image > Default
        target_w, target_h = 1024, 1024
        
        if latent_image is not None:
            target_h = latent_image["samples"].shape[2] * 8
            target_w = latent_image["samples"].shape[3] * 8
        elif image is not None:
            target_h, target_w = image.shape[1], image.shape[2]

        # IMAGE CONDITIONING LOGIC
        input_pil = None
        if image is not None:
            batch_data = 255. * image[0].cpu().numpy()
            input_pil = Image.fromarray(np.clip(batch_data, 0, 255).astype(np.uint8)).convert("RGB")
            
            # Auto-resize input image if latent_image provided a different target resolution
            if input_pil.size != (target_w, target_h):
                print(f"VIBE: Resizing input image to match latent dimensions: {target_w}x{target_h}")
                input_pil = input_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"VIBE: Processing '{positive}' at {target_w}x{target_h}")

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

        final_pil = result if isinstance(result, Image.Image) else result[0]
        final_pil = final_pil.convert("RGB")
        
        if final_pil.size != (target_w, target_h):
            final_pil = final_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
            
        out_tensor = np.array(final_pil).astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(out_tensor)[None, ]

        return (out_tensor,)

NODE_CLASS_MAPPINGS = { "VIBE_Editor": VibeEditorNode }
NODE_DISPLAY_NAME_MAPPINGS = { "VIBE_Editor": "VIBE Image Editor" }
