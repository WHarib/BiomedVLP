import io, subprocess, sys, torch, tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from transformers import AutoTokenizer, AutoModel
from PIL import Image

MODEL_ID = "microsoft/BiomedVLP-BioViL-T"
DEVICE = torch.device("cpu")

try:
    import health_multimodal  # noqa: F401
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "hi-ml-multimodal>=0.2.1", "pydicom", "opencv-python-headless"
    ])

from health_multimodal.image.utils import get_image_inference

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
text_model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).eval()
image_engine = get_image_inference("biovil_t")

# Shared buffer for the current session
buffer = {"image": None}

@torch.no_grad()
def _text_emb(sentence: str) -> torch.Tensor:
    toks = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    if "token_type_ids" in toks:
        toks.pop("token_type_ids")
    return text_model.get_projected_text_embeddings(**toks).squeeze(0)

@torch.no_grad()
def _image_emb(pil_img: Image.Image) -> torch.Tensor:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        pil_img.save(tmp.name)
        tmp.flush()
        emb = image_engine.get_projected_global_embedding(Path(tmp.name))
    return emb

app = FastAPI(docs_url="/docs")

@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return (
        "<h2>BioViL-T multimodal embedding API</h2>"
        "<ul>"
        "<li><code>POST /embed_image</code> – image (waits for text, stores embedding)</li>"
        "<li><code>POST /embed_text</code> – text (returns similarity if image already received, then resets)</li>"
        "</ul>"
    )

@app.post("/embed_image")
async def embed_image(file: UploadFile = File(...)):
    global buffer
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")
    vec = _image_emb(pil)
    buffer["image"] = vec
    return JSONResponse({"status": "image received, waiting for text"})

@app.post("/embed_text")
async def embed_text(text: str = Form(...)):
    global buffer
    text = text.strip()
    if not text:
        raise HTTPException(400, "Empty text prompt.")
    if buffer["image"] is None:
        return JSONResponse({"status": "waiting for image"})
    text_vec = _text_emb(text)
    score = torch.nn.functional.cosine_similarity(buffer["image"], text_vec, dim=0).item()
    # Clear the buffer for next use
    buffer["image"] = None
    return JSONResponse({"cosine_similarity": score})
