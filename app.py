from fastapi import FastAPI, UploadFile, File, Form
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
        "<p>POST an image and optional query to <code>/extract</code> for embedding extraction.</p>"
    )

@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    query: str = Form(None)  # Make the query optional; use `Form(...)` if required
):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # Use query if provided (adapt logic as per your requirement)
        if query:
            features = pipe(image, query)
        else:
            features = pipe(image)

        # features: list [1][sequence_len][hidden_dim], process as needed
        return JSONResponse({
            "query_used": query,
            "shape": [len(features), len(features[0]), len(features[0][0])],
            "features": features[0][0][:10]  # only first 10 features of first token
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
