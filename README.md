# ComfyUI-VIBE 🎨

<div align="left">
  <img src="examples/I2I-woman-in-a-hat.png" width="300" align="left" style="margin-right: 20px;" alt="VIBE Poster">
</div>

Implementation of **VIBE** (Visual Instruction Based Editor) as a custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Edit images using natural language instructions (e.g., "make it winter", "change the dog to a cat").

VIBE leverages the efficient **Sana1.5-1.6B** diffusion model and **Qwen3-VL-2B-Instruct** for fast, high-quality image manipulation.

## ✨ Features
- **Instruction-based Editing**: No complex prompting required.
- **Latent Support**: Connect an `Empty Latent Image` to define output resolution.
- **Fast Inference**: Powered by Sana1.5's linear attention.
- **Local Model Support**: Runs entirely on your hardware.

<br clear="left"/>

## 🖼️ Example Workflow

Drag and drop this image into ComfyUI to load the workflow:

![ComfyUI VIBE Workflow](examples/workflow.png)

---

## 🚀 Installation

1. **Clone the repository**:
   Navigate to your `ComfyUI/custom_nodes` folder and run:
   ```bash
   git clone https://github.com/ato-zen/ComfyUI-VIBE
   ```

2. **Install dependencies**:
   Open terminal in the `ComfyUI-VIBE` folder and run:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📂 Model Setup

This node looks for models in: `ComfyUI/models/vibe/`

1. **Create the target folder**:
   ```bash
   mkdir -p models/vibe
   cd models/vibe
   ```

2. **Download the weights**:
   Clone from Hugging Face (requires [git-lfs](https://git-lfs.github.com/)):
   ```bash
   git clone https://huggingface.co/iitolstykh/VIBE-Image-Edit
   ```

Structure should look like:
```text
📂 ComfyUI/
└── 📂 models/
    └── 📂 vibe/
         └── 📂 VIBE-Image-Edit/
              ├── model_index.json    
              ├── 📂 scheduler/
              ├── 📂 text_encoder/
              ├── 📂 tokenizer/
              ├── 📂 transformer/
              └── 📂 vae/
```

---

## 📜 Credits & Acknowledgements

- **Original Project**: [VIBE: Visual Instruction Based Editor](https://github.com/ai-forever/VIBE)
- **Model Authors**: Grigorii Alekseenko, Aleksandr Gordeev, Irina Tolstykh, et al.
- **Based on**: [Sana](https://github.com/NVlabs/Sana) and [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL).

**ComfyUI Node implementation by [ato-zen](https://github.com/ato-zen).**

---
*License: Apache 2.0*
