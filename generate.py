import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from generator import Generator, LATENT_SIZE

device = 'cuda'

img_channels = 3

generator = Generator()

generator.to(device)

generator.load_state_dict(torch.load('./versions/generator39.pt'))
generator.eval()

n_images_to_generate = 5
output_image_height = 100
output_image_width = 100

noise = torch.randn(n_images_to_generate, LATENT_SIZE).to(device)

# 5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald
cond = torch.tensor([[1, 0, 0, 0, 0]])
# Bangs Big_Lips Big_Nose Black_Hair Blond_Hair
cond1 = torch.tensor([[0, 0, 0, 0, 0]])
# Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin
cond2 = torch.tensor([[0, 0, 0, 0, 0]])
# Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones
cond3 = torch.tensor([[0, 0, 0, 0, 1]])
# Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard
cond4 = torch.tensor([[1, 1, 0, 0, 1]])
# Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks
cond5 = torch.tensor([[1, 0, 0, 0, 0]])
# Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings
cond6 = torch.tensor([[0, 1, 0, 0, 0]])
# Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young
cond7 = torch.tensor([[0, 0, 0, 0, 1]])

cond = torch.cat((cond, cond1, cond2, cond3, cond4, cond5, cond6, cond7), dim=1)
cond = cond.repeat(n_images_to_generate, 1)
cond = cond.to(device)

with torch.no_grad():
    generated_images = generator(noise, cond)


normalized_images = (generated_images * 0.5 + 0.5).clamp(0, 1).cpu().numpy()

plt.figure(figsize=(10, 10))
for i in range(n_images_to_generate):
    plt.subplot(1, n_images_to_generate, i+1)
    
    img_to_display = normalized_images[i].transpose(1, 2, 0)

    if img_to_display.shape[0] != output_image_height or img_to_display.shape[1] != output_image_width:
        from PIL import Image
        img_to_display = Image.fromarray((img_to_display * 255).astype(np.uint8))
        img_to_display = img_to_display.resize((output_image_width, output_image_height), Image.Resampling.LANCZOS)
        img_to_display = np.array(img_to_display) / 255.0

    plt.imshow(img_to_display)
    plt.axis('off')
plt.suptitle(f"Images", fontsize=16)
plt.show()