from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torch
import io

app = FastAPI(docs_url="/docs")

# Load model and tokenizer with remote trust
model = AutoModel.from_pretrained("microsoft/BiomedVLP-BioViL-T", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-BioViL-T", trust_remote_code=True)

@app.get("/", response_class=HTMLResponse)
async def root():
    return (
        "<h1>BioViL-T Feature Extraction API – running</h1>"
        "<p>POST an image and optional query to <code>/extract</code>.</p>"
    )

@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    query: str = Form("No pleural effusion or pneumothorax is seen.")
):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # Tokenise query and remove token_type_ids
        tokenised = tokenizer(query, return_tensors="pt")
        tokenised.pop("token_type_ids", None)  # 🚨 FIX: Remove unsupported key

        with torch.no_grad():
            # Extract embeddings
            text_emb = model.get_projected_text_embeddings(**tokenised)
            image_emb = model.get_projected_image_embeddings(images=[image])

            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(text_emb, image_emb).item()

        return JSONResponse({
            "query_used": query,
            "similarity_score": round(similarity, 4)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

