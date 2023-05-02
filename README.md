## <div align="center">MLP Mixer</div>

This repo focus implement the paper with Pytorch [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/pdf/2105.01601.pdf).

<a align="center" href="https://arxiv.org/pdf/2105.01601.pdf" target="_blank">
      <img width="100%" src=""></a>

## <div align="center">Documentation</div>

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)

```bash
git clone https://github.com/pdaie/mlp-mixer-pytorch  # clone
cd mlp-mixer-pytorch
pip install -r requirements.txt  # install
```

</details>

<details>
<summary>Inference</summary>

```python
import torch
from PIL import Image
from torchvision import transforms
from model.mlp_mixer import MLPMixer

# Model
model = MLPMixer(
  num_classes=2,
  image_size=(300, 300),
  patch_size=100,
  num_mlp_blocks=8,
  projection_dim=512, 
  token_mixing_dim=2048,
  channel_mixing_dim=256
)

model.load_state_dict(torch.load('runs/exp_*/last.pt))

# Image
image_path = "name_image.jpg"
image = Image.open(image_path)

# Transforms
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

image = transform(image)

# Inference
logis = model(image.unsqueeze(0))

# Results
results = logis.argmax(dim=1)
```

<details>
<summary>Training</summary>

The commands below download dog cat dataset automatically and train on it. If you want to train with your dataset, you must to change infomation on file config:
- Set download: False or delete this row.
- Setup dataset follow this structure:

```
train/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
...class_c/
......c_image_1.jpg
......c_image_2.jpg
```

```
val/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
...class_c/
......c_image_1.jpg
......c_image_2.jpg
```

```bash
python train.py --epochs 300 --learning-rate 1e3 --batch-size 128 --image-size 300 --patch-size 100 --num-mlp-blocks 8 --projection-dim 512 --token-mixing-dim 2048 --channel-mixing-dim 256 --num-workers 1 --device cuda:0
                                                               
```

</details>
