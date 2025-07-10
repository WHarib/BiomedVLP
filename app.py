from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
from transformers import AutoTokenizer, AutoModel, pipeline
import io, torch

app = FastAPI(docs_url="/docs")

# ────────────────────────────────────────────────────────────
# Load tokenizer + model once at start-up
# ────────────────────────────────────────────────────────────
MODEL_ID = "microsoft/BiomedVLP-BioViL-T"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model     = AutoModel.from_pretrained(MODEL_ID,    trust_remote_code=True)

pipe = pipeline(
    "feature-extraction",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=None,           # no image preprocessor config in repo
)

@app.get("/", response_class=HTMLResponse)
async def root():
    return (
        "<h1>BioViL-T Embedding API</h1>"
        "<ul>"
        "<li>POST <code>file</code> only → image embedding</li>"
        "<li>POST <code>query</code> only → text embedding</li>"
        "<li>POST both → joint embedding</li>"
        "</ul>"
    )

# ────────────────────────────────────────────────────────────
# /extract endpoint
# ────────────────────────────────────────────────────────────
@app.post("/extract")
async def extract(
    file:  UploadFile | None = File(None),   # make both optional
    query: str        | None = Form(None)
):
    """
    • Image only   → pipe(image)
    • Text only    → pipe(text)
    • Image + text → pipe({"image": image, "text": text})
    """

    if file is None and query is None:
        raise HTTPException(400, "Supply at least an image or a query")

    # ---------- normalise QUERY ----------
    if isinstance(query, bytes):
        query = query.decode()
    if isinstance(query, list):
        query = query[0] if query else None
    query = query.strip() if query else None
    # -------------------------------------

    # ---------- normalise IMAGE ----------
    img = None
    if file is not None:
        try:
            img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"Bad image: {e}")
    # -------------------------------------

    # Build input for pipeline()
    if img is not None and query:
        pipe_input = {"image": img, "text": query}
    elif img is not None:
        pipe_input = img
    else:  # text only
        pipe_input = query

    # Run inference
    with torch.no_grad():
        feats = pipe(pipe_input)

    # Prepare quick summary
    shape = [len(feats), len(feats[0]), len(feats[0][0])]

    return JSONResponse(
        {
            "image_supplied": img is not None,
            "query_supplied": query,
            "shape": shape,
            "preview": feats[0][0][:10],  # first 10 dims of CLS token
        }
    )
