# app.py  ───────────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from PIL import Image
import torch, io, os

MODEL_ID   = "microsoft/BiomedVLP-BioViL-T"
REVISION   = "a3e25dcc5c11ee95e845cd9cfa66f7f0043744f4"     # pin one stable commit
DEVICE     = "cpu"                                           # Spaces usually default to CPU

# ── Load everything once at start-up ───────────────────────────────
tokenizer  = AutoTokenizer.from_pretrained(
                MODEL_ID, revision=REVISION, trust_remote_code=True
            )
processor  = AutoProcessor.from_pretrained(
                MODEL_ID, revision=REVISION, trust_remote_code=True
            )
model      = AutoModel.from_pretrained(
                MODEL_ID, revision=REVISION, trust_remote_code=True
            ).eval()

# ── FastAPI instance and landing page ─────────────────────────────
app = FastAPI(docs_url="/docs")

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (
        "<h2>BioViL-T multimodal embedding API</h2>"
        "<ul>"
        "<li><code>POST /embed_text</code> &nbsp;→ text → 512-d vector</li>"
        "<li><code>POST /embed_image</code> → image → 512-d vector</li>"
        "<li><code>POST /similarity</code> &nbsp;→ image + text → cosine score</li>"
        "</ul>"
    )

# ── Helper functions ──────────────────────────────────────────────
def _text_embedding(text: str) -> torch.Tensor:
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        return model.get_projected_text_embeddings(**toks).squeeze(0)   # (512,)

def _image_embedding(pil_img: Image.Image) -> torch.Tensor:
    pix = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        return model.get_projected_image_embeddings(**pix).squeeze(0)   # (512,)

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
    file : UploadFile = File(...),
    text : str        = Form(...)
):
    pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_vec  = _image_embedding(pil)
    text_vec = _text_embedding(text.strip())
    score    = torch.nn.functional.cosine_similarity(img_vec, text_vec, dim=0).item()
    return JSONResponse({"cosine_similarity": score})
