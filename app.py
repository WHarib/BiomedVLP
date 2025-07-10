from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
from PIL import Image
import torch, io

MODEL_ID = "microsoft/BiomedVLP-BioViL-T"
DEVICE = "cpu"  # Hugging Face Spaces default

# Load model, tokenizer, and image processor once at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).eval()

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

def _text_embedding(text: str) -> torch.Tensor:
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        return model.get_projected_text_embeddings(**toks).squeeze(0)

def _image_embedding(pil_img: Image.Image) -> torch.Tensor:
    pix = image_processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        return model.get_projected_image_embeddings(**pix).squeeze(0)

@app.post("/embed_text")
async def embed_text(text: str = Form(...)):
    text = text.strip()
    if not text:
        raise HTTPException(400, "Empty text prompt.")
    vec = _text_embedding(text)
    return JSONResponse({"embedding": vec.tolist()})

@app.post("/embed_image")
async def embed_image(file: UploadFile = File(...)):
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")
    vec = _image_embedding(pil)
    return JSONResponse({"embedding": vec.tolist()})

@app.post("/similarity")
async def similarity(file: UploadFile = File(...), text: str = Form(...)):
    pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_vec = _image_embedding(pil)
    text_vec = _text_embedding(text.strip())
    score = torch.nn.functional.cosine_similarity(img_vec, text_vec, dim=0).item()
    return JSONResponse({"cosine_similarity": score})
