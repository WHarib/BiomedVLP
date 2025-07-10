# app.py ───────────────────────────────────────────────────────────
"""
BioViL-T multimodal embedding API
--------------------------------
Routes
  • POST /embed_text   → returns a 512-d text embedding
  • POST /embed_image  → returns a 512-d image embedding
  • POST /similarity   → returns cosine similarity (image ↔︎ text)
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from PIL import Image
import torch, io

MODEL_ID = "microsoft/BiomedVLP-BioViL-T"   # latest repo head
DEVICE   = "cpu"                            # Hugging Face Spaces default

# ── Load model, tokenizer, processor once at start-up ─────────────
tokenizer  = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)  # custom CXRBertTokenizer is loaded via `auto_map`

processor  = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model      = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
).eval()

# ── FastAPI instance & landing page ───────────────────────────────
app = FastAPI(docs_url="/docs")

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (
        "<h2>BioViL-T multimodal embedding API (CPU)</h2>"
        "<ul>"
        "<li><code>POST /embed_text</code> – text → 512-d vector</li>"
        "<li><code>POST /embed_image</code> – image → 512-d vector</li>"
        "<li><code>POST /similarity</code> – image + text → cosine score</li>"
        "</ul>"
    )

# ── Helper functions ──────────────────────────────────────────────
def _text_embedding(text: str) -> torch.Tensor:
    """Return a (512,) tensor for a single sentence."""
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        return model.get_projected_text_embeddings(**toks).squeeze(0)

def _image_embedding(pil_img: Image.Image) -> torch.Tensor:
    """Return a (512,) tensor for a single chest X-ray image."""
    pix = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        return model.get_projected_image_embeddings(**pix).squeeze(0)

# ── /embed_text ───────────────────────────────────────────────────
@app.post("/embed_text")
async def embed_text(text: str = Form(...)):
    text = text.strip()
    if not text:
        raise HTTPException(400, "Empty text prompt.")
    vec = _text_embedding(text)
    return JSONResponse({"embedding": vec.tolist()})

# ── /embed_image ──────────────────────────────────────────────────
@app.post("/embed_image")
async def embed_image(file: UploadFile = File(...)):
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")
    vec = _image_embedding(pil)
    return JSONResponse({"embedding": vec.tolist()})

# ── /similarity ───────────────────────────────────────────────────
@app.post("/similarity")
async def similarity(
    file: UploadFile = File(...),
    text: str       = Form(...)
):
    pil       = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_vec   = _image_embedding(pil)
    text_vec  = _text_embedding(text.strip())
    score     = torch.nn.functional.cosine_similarity(img_vec, text_vec, dim=0).item()
    return JSONResponse({"cosine_similarity": score})
