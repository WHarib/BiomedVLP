from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
from transformers import pipeline, AutoFeatureExtractor, AutoTokenizer, AutoModel
import io

app = FastAPI(docs_url="/docs")

# Explicitly load feature extractor and tokenizer
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "microsoft/BiomedVLP-BioViL-T", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/BiomedVLP-BioViL-T", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "microsoft/BiomedVLP-BioViL-T", trust_remote_code=True
)

# Build a custom pipeline
pipe = pipeline(
    task="feature-extraction",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
)

@app.get("/", response_class=HTMLResponse)
async def root():
    return ("<h1>BioViL-T Feature Extraction API – running</h1>"
            "<p>POST an image and optional query to <code>/extract</code>.</p>")

@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    query: str = Form(None)
):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        if query:
            inputs = {"image": img, "text": query}
        else:
            inputs = img

        features = pipe(inputs)
        return JSONResponse({
            "query_used": query,
            "shape": [len(features), len(features[0]), len(features[0][0])],
            "features": features[0][0][:10]
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
