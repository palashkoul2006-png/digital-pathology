# Digital Pathology AI — Architecture Diagram

## System Overview

A full-stack AI-powered skin cancer analysis web application using a fine-tuned Vision Transformer (ViT), a Stable Diffusion generative engine, and a Gemini LLM agentic reporting layer.

---

## High-Level Architecture

```mermaid
flowchart TD
    subgraph CLIENT["🌐 Frontend (Browser)"]
        UI["index.html\n─────────────\nFile Upload / Camera Capture\nResults Dashboard\nHeatmap Viewer\nChat Interface\nPatient History Chart"]
        JS["script.js\n─────────────\nAPI calls (fetch)\nChart.js rendering\nBase64 image handling"]
        CSS["style.css + background.mp4\n─────────────\nAnimated UI / Dark theme"]
    end

    subgraph FLASK["⚙️ Flask Backend — app.py"]
        R1["POST /predict"]
        R2["POST /api/simulate_progression"]
        R3["POST /api/chat"]
        R4["GET  /api/history/<patient_id>"]
        R5["GET  /  → index.html"]
    end

    subgraph INFERENCE["🧠 AI Inference Pipeline"]
        PP["utils.py → preprocess_image()\n• OpenCV denoising\n• LANCZOS resize → 224×224\n• Normalize [0,1]"]
        DT["utils.py → detect_image_type()\n• HSV hue analysis (H&E stain)\n• Saturation variance\n• Texture entropy\n→ 'pathology' or 'external'"]
        VIT["models.py → _ViTWrapper\n• google/vit-base-patch16-224\n• Fine-tuned on HAM10000\n• 7-class skin cancer classifier\n• 98.52% accuracy\n• CUDA / CPU inference"]
        HEAT["heatmaps.py → generate_heatmap()\n1. ViT Attention Rollout (preferred)\n2. Grad-CAM (TF CNN fallback)\n3. Simulated edge-energy (last resort)\n→ base64 JPEG overlay"]
    end

    subgraph GEN["🎨 Generative Engine"]
        SD["Stable Diffusion v1.5\n(safetensors base model)"]
        LORA["PathAI LoRA\npathai_melanoma_v1.safetensors\n• Custom trained on melanoma\n• strength=0.65\n• Img2Img pipeline"]
        VRAM["RTX 3050 6GB\nenable_model_cpu_offload()"]
    end

    subgraph LLM["🤖 Agentic AI Layer — utils.py"]
        GEMINI["Gemini 2.5 Flash\n(Google GenAI SDK)\n• generate_agentic_report()\n• chat_with_agent()\n→ JSON medical report\n→ Conversational Q&A"]
    end

    subgraph DB["🗄️ Database — MongoDB"]
        MONGO["pathai_db.patients\n• patient_id\n• date (UTC)\n• image_path\n• prediction\n• confidence\n• risk_level"]
    end

    subgraph MODELS["📦 Saved Model — ./saved_model"]
        CFG["config.json"]
        WEIGHTS["model.safetensors\n(~327 MB)"]
        PROC["preprocessor_config.json"]
    end

    subgraph TRAINING["🏋️ Training Pipeline (offline)"]
        DS["HAM10000 Dataset\n(marmal88/skin_cancer @ HuggingFace)\n7 classes: akiec, bcc, bkl,\ndf, mel, nv, vasc"]
        TRAINPY["train.py\n• ViTForImageClassification\n• Fine-tune on 7 classes\n→ save_pretrained('./saved_model')"]
        FT["finetune.py\n• LoRA fine-tuning\n• Melanoma progression data"]
        PREP["prep_dataset.py\n• Dataset preparation"]
    end

    %% Client ↔ Flask
    UI -- "multipart/base64 POST" --> R1
    UI -- "file POST" --> R2
    UI -- "JSON POST" --> R3
    UI -- "GET" --> R4
    JS -- renders --> UI

    %% Flask /predict pipeline
    R1 --> PP
    R1 --> DT
    PP --> VIT
    DT --> VIT
    VIT --> HEAT
    VIT --> GEMINI
    VIT --> MONGO
    HEAT -- "heatmap_b64" --> R1
    GEMINI -- "recommendation JSON" --> R1
    R1 -- "JSON response" --> UI

    %% Flask /simulate
    R2 --> SD
    SD --> LORA
    LORA --> VRAM
    VRAM -- "synthetic_image b64" --> R2
    R2 -- "JSON response" --> UI

    %% Flask /chat
    R3 --> GEMINI

    %% Flask /history
    R4 --> MONGO

    %% Model loading
    MODELS --> VIT

    %% Training feeds model
    DS --> TRAINPY
    TRAINPY --> MODELS
    PREP --> DS
    FT --> LORA
```

---

## Component Breakdown

### 1. Frontend (`/templates` + `/static`)

| File | Role |
|---|---|
| `index.html` | Single-page app — upload, results, heatmap, chat, history |
| `script.js` | All client-side logic: API calls, Chart.js probability bars, camera capture |
| `style.css` | Dark-mode glassmorphism UI, animations |
| `background.mp4` | Animated video background |

---

### 2. Flask Backend (`app.py`)

| Route | Method | Description |
|---|---|---|
| `/` | GET | Serves `index.html` |
| `/predict` | POST | Main inference endpoint — file or base64 |
| `/api/simulate_progression` | POST | Stable Diffusion LoRA image-to-image |
| `/api/chat` | POST | Gemini-powered conversational Q&A |
| `/api/history/<patient_id>` | GET | Patient record fetch from MongoDB |

---

### 3. AI Inference Pipeline

```mermaid
flowchart LR
    IMG["Uploaded Image"] --> PRE["preprocess_image()\nDenoise → Resize → Normalize"]
    IMG --> DET["detect_image_type()\nHSV + Entropy → 'pathology'/'external'"]
    PRE --> VIT2["ViT Fine-tuned\n7-class Softmax"]
    DET --> VIT2
    VIT2 --> OUT["top_class, confidence\nrisk_level, all_probs"]
    OUT --> HMAP["Attention Rollout\n14×14 grid → JET colormap overlay"]
    OUT --> RPT["Gemini LLM\nJSON medical report"]
```

---

### 4. ViT Model (`models.py` + `./saved_model`)

| Detail | Value |
|---|---|
| **Base Model** | `google/vit-base-patch16-224` |
| **Dataset** | HAM10000 (7-class skin cancer) |
| **Accuracy** | 98.52% |
| **Input** | 224×224 RGB, normalized `mean=0.5, std=0.5` |
| **Output** | 7-class softmax probabilities |
| **Hardware** | CUDA (RTX 3050) / CPU fallback |
| **Classes** | Actinic Keratoses, Basal Cell Carcinoma, Benign Keratosis, Dermatofibroma, Melanocytic Nevi, Melanoma, Vascular Lesions |

---

### 5. Heatmap Generation (`heatmaps.py`)

```mermaid
flowchart TD
    H1{"Has .model, .processor, .device?"}
    H1 -- Yes --> AR["ViT Attention Rollout\nAbnar & Zuidema 2020\nCLS→patch 14×14 grid"]
    H1 -- No --> H2{"Has Conv2D layers?"}
    H2 -- Yes --> GC["Grad-CAM\nLast Conv2D layer gradients"]
    H2 -- No --> SE["Simulated Edge-Energy\nLaplacian + Gaussian blend"]
    AR --> BLEND["Resize → JET colormap → 50/50 blend with original"]
    GC --> BLEND
    SE --> BLEND
    BLEND --> B64["base64 JPEG data URL"]
```

---

### 6. Generative Engine (`app.py` + `PathAI_LoRA/`)

| Detail | Value |
|---|---|
| **Base Model** | Stable Diffusion v1.5 (safetensors) |
| **LoRA Weights** | `pathai_melanoma_v1.safetensors` |
| **Pipeline** | `StableDiffusionImg2ImgPipeline` |
| **Strength** | 0.65 (preserves skin structure, adds melanoma texture) |
| **Input** | Any benign skin image (resized to 512×512) |
| **VRAM Strategy** | `enable_model_cpu_offload()` for RTX 3050 6GB |

---

### 7. Agentic AI Layer (`utils.py`)

| Function | Model | Output |
|---|---|---|
| `generate_agentic_report()` | Gemini 2.5 Flash | JSON: `medical_advice`, `preventive_routine`, `lifestyle_suggestions`, `maps_link`, `disclaimer` |
| `chat_with_agent()` | Gemini 2.5 Flash | Plain-text conversational response with diagnostic context |

---

### 8. Database (`MongoDB`)

| Collection | Fields |
|---|---|
| `pathai_db.patients` | `patient_id`, `date`, `image_path`, `prediction`, `confidence`, `risk_level` |

---

### 9. Training Pipeline (Offline)

| Script | Purpose |
|---|---|
| `train.py` | Fine-tune `google/vit-base-patch16-224` on HAM10000 → saves to `./saved_model` |
| `finetune.py` | LoRA fine-tuning for Stable Diffusion on melanoma images |
| `prep_dataset.py` | Dataset preparation and formatting |
| `ImportModel.py` | Model import utilities |

---

## Data Flow Summary

```mermaid
sequenceDiagram
    participant U as User (Browser)
    participant F as Flask (app.py)
    participant V as ViT Model
    participant H as Heatmap Engine
    participant G as Gemini LLM
    participant M as MongoDB

    U->>F: POST /predict (image)
    F->>F: preprocess_image() + detect_image_type()
    F->>V: predict_image()
    V-->>F: prediction, confidence, risk, all_probs
    F->>H: generate_heatmap()
    H-->>F: base64 heatmap JPEG
    F->>G: generate_agentic_report()
    G-->>F: JSON medical report
    F->>M: Insert patient record
    F-->>U: JSON (prediction + heatmap + report)
```

---

> **Tech Stack Summary:** Python · Flask · HuggingFace Transformers · PyTorch · Stable Diffusion (diffusers) · Google Gemini SDK · MongoDB · OpenCV · Vanilla JS · CSS3 · Chart.js
