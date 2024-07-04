
# Anime Image Generator

Generate unconditional anime portrait images.

# Sample usage

```python
import torch
from diffusers import DiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = DiffusionPipeline.from_pretrained("FatRat7/anime-image-generator", use_safetensors=True).to(device)

image = pipeline(num_inference_steps=1000).images[0]
image.save("path")
```

# Dataset
Dataset https://huggingface.co/datasets/CaptionEmporium/anime-caption-danbooru-2021-sfw-5m-hq.
100,000 images of men and women were used to train the model.

# Code Info
Due to problems with training the model with the help of the pytorch lighting library I decided to use pure pytorch.
All files with lighting prefix are not used in the final model. When I find the source of the problem I will go back
to using lighting library.
