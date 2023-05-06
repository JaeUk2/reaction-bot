from diffusers import StableDiffusionPipeline
import torch

model_path = "/opt/ml/diffusion/emoji-stable-diffusion-finetuned-lora"
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "[sad] I'm in a bad mood because I messed up the test"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("pokemon.png")