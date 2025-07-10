from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
from transformers import pipeline, AutoTokenizer, AutoModel
import io

app = FastAPI(docs_url="/docs")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-BioViL-T", trust_remote_code=True)
model = AutoModel.from_pretrained("microsoft/BiomedVLP-BioViL-T", trust_remote_code=True)

# Build a pipeline
pipe = pipeline(
    "feature-extraction",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=None,
)

@app.get("/", response_class=HTMLResponse)
async def root():
    return (
        "<h1>BioViL-T Feature Extraction API – running</h1>"
        "<p>POST an image and optional query to <code>/extract</code>.</p>"
    )

@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    query: str = Form(None)
):
    try:
        # Load the image
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # Build the input dictionary
        inputs = {"image": img}
        if query and isinstance(query, str):
            inputs["text"] = query.strip()

        # Run through the pipeline
        features = pipe(inputs)

        return JSONResponse({
            "query_used": inputs.get("text", None),
            "shape": [
                len(features), len(features[0]),
                len(features[0][0])
            ],
            "features": features[0][0][:10]  # preview of the embedding
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
