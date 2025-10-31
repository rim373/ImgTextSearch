from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

# ==================== LOAD .ENV ====================
load_dotenv()  # Load environment variables from .env file
# ==================== CONFIG ====================
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
INDEX_NAME = os.getenv("INDEX_NAME", "test_index")
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", 20))
DATA_PATH = os.getenv("DATA_PATH", "./data")  # Default to ./data if not set

# ==================== FASTAPI APP ====================
app = FastAPI(title="Image & Tag Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (images) via /images/ URL
app.mount("/images", StaticFiles(directory=DATA_PATH), name="images")

# ==================== ELASTICSEARCH ====================
es = Elasticsearch(ES_HOST, request_timeout=600)
if not es.ping():
    raise ConnectionError("❌ Elasticsearch not reachable")

# ==================== VGG16 MODEL ====================
model = VGG16(weights="imagenet", include_top=False, pooling="avg")

def extract_vgg_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x, verbose=0)
        return features.flatten().tolist()
    except Exception as e:
        print(f"⚠️ Error processing {img_path}: {e}")
        return []

# ==================== POST: Text Search ====================
@app.post("/search/text")
async def search_text(query_text: str = Form(...), top_k: int = TOP_K_DEFAULT):
    try:
        query = {
            "size": top_k,
            "query": {
                "match": {
                    "tag": {"query": query_text, "fuzziness": "AUTO"}
                }
            }
        }
        res = es.search(index=INDEX_NAME, body=query)

        results = []
        for doc in res["hits"]["hits"]:
            local_path = doc["_source"]["path"]
            filename = os.path.basename(local_path)
            # Path relative to DATA_PATH
            web_path = f"all_images_final/{filename}"
            results.append({"path": web_path})

        return {"results": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ==================== POST: Image Search ====================
@app.post("/search/image")
async def search_image(file: UploadFile = File(...), top_k: int = TOP_K_DEFAULT):
    try:
        img_bytes = await file.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        temp_path = "temp_query.jpg"
        img.save(temp_path)

        emb = extract_vgg_features(temp_path)
        if not emb or len(emb) != 512:
            return JSONResponse(status_code=400, content={"error": "Invalid image embedding"})

        query = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": emb}
                    }
                }
            }
        }

        res = es.search(index=INDEX_NAME, body=query)

        results = []
        for doc in res["hits"]["hits"]:
            local_path = doc["_source"]["path"]
            filename = os.path.basename(local_path)
            # Path relative to DATA_PATH
            web_path = f"all_images_final/{filename}"
            results.append({"path": web_path})

        return {"results": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ==================== RUN ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)