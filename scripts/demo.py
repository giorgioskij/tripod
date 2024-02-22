import torch
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python demo.py <path_to_image>")
    sys.exit(1)

import script_config
from kolnet import Kolnet

checkpoint_path = Path("checkpoints/tripod_v1.ckpt")
if not checkpoint_path.exists():
    ans = input(
        "Can't find checkpoint locally: do you want to donwload it? [Y/n]: ")
    if ans.lower() in ["n", "no"]:
        sys.exit(1)
    print("Downloading checkpoint...")

    import gdown

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://drive.google.com/uc?id=1uZuZ-l-OknsdAqzZ789SKgDvSVBWVyJ5'
    output = 'checkpoints/tripod_v1.ckpt'
    gdown.download(url, output, quiet=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = Path(sys.argv[1])
if not image_path.exists():
    print(f"Can't find image at {image_path}")
    sys.exit(1)

model: Kolnet = Kolnet.load_from_checkpoint(checkpoint_path).eval().cuda()
image = to_tensor(Image.open(str(image_path)).convert("RGB"))

print("Processing image...")
with torch.no_grad():
    output = model.sharpen(image.to(model.device))

new_name = image_path.stem + "_sharpened" + image_path.suffix
new_path = image_path.parent / new_name
save_image(output, new_path)
