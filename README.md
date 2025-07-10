---
title: BioViL-T Feature Extraction API
emoji: 🧬
colorFrom: indigo
colorTo: gray
sdk: docker
sdk_version: "1.0"
app_file: app.py
pinned: false
---

# BioViL-T Feature Extraction API

A Hugging Face Space that exposes the `microsoft/BiomedVLP-BioViL-T` model as a FastAPI service for feature extraction from medical images.

## Endpoint

**POST** `/extract`

- **Input:** `file` (PNG/JPG medical image)
- **Output:** Example of the feature embedding for the first token

## Example usage

```python
import requests
with open("image.png", "rb") as f:
    resp = requests.post(
        "https://yourusername-biovil-t-demo.hf.space/extract",
        files={"file": f}
    )
    print(resp.json())
