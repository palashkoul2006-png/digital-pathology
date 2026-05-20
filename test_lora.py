import torch
from diffusers import StableDiffusionPipeline

print("Loading Base Model...")
# 1. Load your downloaded offline base model directly into your RTX 3050
pipe = StableDiffusionPipeline.from_single_file(
    r"C:\AI_Training\base_model\v1-5-pruned-emaonly.safetensors",
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

print("Injecting Clinical LoRA...")
# 2. Attach your newly trained neural weights
pipe.load_lora_weights(r"C:\Users\palas\OneDrive - Vishwakarma Institute of Technology\Desktop\digital_pathology\PathAI_LoRA\model\pathai_melanoma_v1.safetensors")

print("Generating Test Image...")
# 3. Generate the image using your trigger word
prompt = "macro clinical photography of a p4th4i melanoma on skin, highly detailed, sharp focus, medical imaging"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

# 4. Save the result
image.save("lora_test_result.png")
print("✅ Success! Check your folder for lora_test_result.png")