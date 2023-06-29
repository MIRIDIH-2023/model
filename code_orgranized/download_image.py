from PIL import Image
import requests
from io import BytesIO
import re
from tqdm.notebook import tnrange
import pandas as pd
import json

df = pd.read_csv("C:/Users/qazxs/Downloads/xml_20230628-1.csv")

def clean_url(url):
    return re.sub(r'(https://file\.miricanvas\.com/).*(template_thumb)', r'\1\2', url)

for i in tnrange(39753, len(df)):
  try:
    image_url = 'thumbnail_url'
    url = df.iloc[i][image_url]
    url = clean_url(url)
    tiff_image =  Image.open(BytesIO(requests.get(url).content))

    png_image_path = f"image/image_{i}.png"
    tiff_image.save(png_image_path, "PNG")
  except Exception as e:
    print(e)
    print(i)
    continue
