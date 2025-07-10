from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
from transformers import pipeline
import io

app = FastAPI(docs_url="/docs")

# Load pipeline at startup
pipe = pipeline("feature-extraction", model="microsoft/BiomedVLP-BioViL-T", trust_remote_code=True)

@app.get("/", response_class=HTMLResponse)
async def root():
    return (
        "<h1>BioViL-T Feature Extraction API – running</h1>"
        "<p>POST an image to <code>/extract</code> for embedding extraction.</p>"
    )

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    features = pipe(image)
    # features: list [1][sequence_len][hidden_dim], so flatten or process as needed
    return JSONResponse({
        "shape": [len(features), len(features[0]), len(features[0][0])],
        "features": features[0][0][:10]  # show only first 10 features of first token for brevity
    })
