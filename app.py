# app.py ────────────────────────────────────────────────────────────
"""
BioViL-T multimodal embedding API (CPU)
--------------------------------------
Routes
  • POST /embed_text   → 512-d text embedding
  • POST /embed_image  → 512-d image embedding
  • POST /similarity   → cosine(image ↔︎ text)
"""

import subprocess, sys, io, torch
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import AutoTokenizer, AutoModel
from PIL import Image

# ───────────────────────────────────────────────────────────────────
MODEL_ID   = "microsoft/BiomedVLP-BioViL-T"
IMG_CKPT   = "biovil_t_image_model_proj_size_128.pt"  # lives in the repo
DEVICE     = torch.device("cpu")                      # Spaces default

# install hi-ml-multimodal on first cold-start (≈ 5-10 s)
try:
    import health_multimodal                            # noqa: F401
except ImportError:                                     # pragma: no cover
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         "hi-ml-multimodal>=0.2.1", "pydicom", "opencv-python-headless"]
    )

from health_multimodal.image import get_biovil_resnet_inference
from health_multimodal.vlp   import ImageTextInferenceEngine

# ── Text encoder (CXRBERT) via 🤗 Transformers ─────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
txt_model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).eval()

def _text_embedding(text: str) -> torch.Tensor:
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        return txt_model.get_projected_text_embeddings(**toks).squeeze(0)  # (512,)

# ── Image encoder via hi-ml (loads ViT-ResNet + projector) ─────────
ckpt_path = Path(__file__).with_name(IMG_CKPT)           # auto-downloaded by HF
img_infer = get_biovil_resnet_inference(str(ckpt_path), device=DEVICE)

def _image_embedding(pil_img: Image.Image) -> torch.Tensor:
    with torch.no_grad():
        return img_infer.get_projected_global_embedding(pil_img).squeeze(0)  # (512,)

# joint engine (not strictly required but handy if you expand later)
joint_engine = ImageTextInferenceEngine(
    image_inference_engine=img_infer,
    text_inference_engine=None  # we handle text ourselves
)

# ── FastAPI app & landing page ─────────────────────────────────────
app = FastAPI(docs_url="/docs")

@app.get("/", response_class=HTMLResponse)
async def index() -> str:  # noqa: D401
    return (
        "<h2>BioViL-T multimodal embedding API (CPU)</h2>"
        "<ul>"
        "<li><code>POST /embed_text</code> – text → 512-d vector</li>"
        "<li><code>POST /embed_image</code> – image → 512-d vector</li>"
        "<li><code>POST /similarity</code> – image + text → cosine score</li>"
        "</ul>"
    )

# ── /embed_text ────────────────────────────────────────────────────
@app.post("/embed_text")
async def embed_text(text: str = Form(...)):
    text = text.strip()
    if not text:
        raise HTTPException(400, "Empty text prompt.")
    vec = _text_embedding(text)
    return JSONResponse({"embedding": vec.tolist()})

# ── /embed_image ───────────────────────────────────────────────────
@app.post("/embed_image")
async def embed_image(file: UploadFile = File(...)):
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")
    vec = _image_embedding(pil)
    return JSONResponse({"embedding": vec.tolist()})

# ── /similarity ────────────────────────────────────────────────────
@app.post("/similarity")
async def similarity(file: UploadFile = File(...), text: str = Form(...)):
    pil      = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_vec  = _image_embedding(pil)
    text_vec = _text_embedding(text.strip())
    score    = torch.nn.functional.cosine_similarity(img_vec, text_vec, dim=0).item()
    return JSONResponse({"cosine_similarity": score})
