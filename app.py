# app.py ─────────────────────────────────────────────────────────────
"""
BioViL-T multimodal embedding API (CPU)

Routes:
  • POST /embed_text   → 512-d text embedding
  • POST /embed_image  → 512-d image embedding
  • POST /similarity   → cosine(image ↔ text)
"""

# stdlib & installs at cold-start
import io, subprocess, sys, torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import AutoTokenizer, AutoModel
from PIL import Image

# ── constants ──────────────────────────────────────────────────────
TEXT_MODEL_ID = "microsoft/BiomedVLP-BioViL-T"   # CXRBERT inside BioViL-T
DEVICE        = torch.device("cpu")              # Space default

# ── install HI-ML if absent (adds ~5-10 s to first cold-start) ─────
try:
    import health_multimodal                              # noqa: F401
except ImportError:                                       # pragma: no cover
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         "hi-ml-multimodal>=0.2.1", "pydicom", "opencv-python-headless"]
    )
from health_multimodal.image.utils import get_image_inference
from health_multimodal.image import ImageModelType

# ── load models once ───────────────────────────────────────────────
tokenizer  = AutoTokenizer.from_pretrained(TEXT_MODEL_ID, trust_remote_code=True)
text_model = AutoModel.from_pretrained(TEXT_MODEL_ID,   trust_remote_code=True).eval()

image_engine = get_image_inference(ImageModelType.BIOVIL_T)  # handles weights + transforms

# ── helper functions ──────────────────────────────────────────────
@torch.no_grad()
def _text_emb(sentence: str) -> torch.Tensor:
    toks = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    return text_model.get_projected_text_embeddings(**toks).squeeze(0)   # (512,)

@torch.no_grad()
def _image_emb(pil_img: Image.Image) -> torch.Tensor:
    return image_engine.get_projected_global_embedding(pil_img)          # (512,)

# ── FastAPI app ────────────────────────────────────────────────────
app = FastAPI(docs_url="/docs")

@app.get("/", response_class=HTMLResponse)
async def root() -> str:   # noqa: D401
    return (
        "<h2>BioViL-T multimodal embedding API</h2>"
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
    vec = _text_emb(text)
    return JSONResponse({"embedding": vec.tolist()})

# ── /embed_image ───────────────────────────────────────────────────
@app.post("/embed_image")
async def embed_image(file: UploadFile = File(...)):
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:                                       # noqa: BLE001
        raise HTTPException(400, f"Bad image: {e}")
    vec = _image_emb(pil)
    return JSONResponse({"embedding": vec.tolist()})

# ── /similarity ────────────────────────────────────────────────────
@app.post("/similarity")
async def similarity(file: UploadFile = File(...), text: str = Form(...)):
    pil  = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_vec  = _image_emb(pil)
    text_vec = _text_emb(text.strip())
    score = torch.nn.functional.cosine_similarity(img_vec, text_vec, dim=0).item()
    return JSONResponse({"cosine_similarity": score})
