from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

# Load the Stable Diffusion pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    use_auth_token=True,
).to("cuda")

# Load the input image
input_image = Image.open("/media/mostafahaggag/Shared_Drive/selfdevelopment/datasets/crops_mvtec_format/yellow/train/good/crop_393.bmp").convert("RGB")
print(type(input_image))  # Should print: <class 'PIL.Image.Image'>

# Define the prompt
prompt = "put black dot on the image"

# Generate the image
generated_image = pipe(prompt=prompt, image=input_image, strength=0.9, guidance_scale=9.5).images[0]

# Save the generated image
generated_image.save("defective_pill.png")