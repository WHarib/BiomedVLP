from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
from transformers import pipeline
import io

app = FastAPI(docs_url="/docs")

# Load the multimodal feature-extraction pipeline
pipe = pipeline("feature-extraction", model="microsoft/BiomedVLP-BioViL-T", trust_remote_code=True)

@app.get("/", response_class=HTMLResponse)
async def root():
    return (
        "<h1>BioViL-T Feature Extraction API – running</h1>"
        "<p>POST an image and optional query to <code>/extract</code> for embedding extraction.</p>"
    )

@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    query: str = Form(None)
):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # If a query is provided, use multimodal input
        if query:
            features = pipe({"image": image, "text": query})
        else:
            features = pipe(image)

        return JSONResponse({
            "query_used": query,
            "shape": [len(features), len(features[0]), len(features[0][0])],
            "features": features[0][0][:10]  # Show first 10 features for brevity
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
