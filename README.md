
# Anime Image Generator

Generate unconditional anime portrait images.

# Sample usage
install `diffusers`

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("model").to("cuda")

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
