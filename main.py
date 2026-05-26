import os
import time
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize the core FastAPI Application
app = FastAPI(
    title="VibeCheck AI - Semantic Vector Search API", 
    version="1.0.0",
    description="Low-latency semantic audio discovery mapping abstract queries to HNSW indexes via vector similarity lookup."
)

# Global placeholders for lazy loading resources efficiently on startup
model = None
collection = None

@app.on_event("startup")
def load_resources():
    """
    Life-cycle hook running automatically on server boot. 
    Loads the sentence transformer model and binds to the persistent vector database.
    """
    global model, collection
    print("🚀 Initializing high-dimensional embedding framework (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("📦 Establishing persistent interface to ChromaDB vector storage...")
    chroma_client = chromadb.PersistentClient(path="./database")
    collection = chroma_client.get_or_create_collection(name="vibe_tracks")
    print("✅ System Core initialized and hot-linked successfully.")


# Mount the static site directories to serve your front-end asset layers
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_home():
    """
    Primary Root endpoint. Renders the sleek, neon glassmorphism presentation layer UI.
    """
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "VibeCheck AI Backend Live. Please populate 'static/index.html' to render the visual deck UI layer."}


class SearchResponse(BaseModel):
    title: str
    artist: str
    match_score: float
    summary: str


@app.get("/api/search", response_model=list[SearchResponse])
def semantic_search(vibe: str = Query(..., description="The abstract feeling, scene, or scenario to analyze")):
    """
    Core Search API Engine. Converts abstract client queries into text embeddings 
    and executes an optimized nearest-neighbor spatial evaluation.
    """
    # Safety fallback if model or collection failed to load on startup
    if model is None or collection is None:
        return []

    # 1. Transform unstructured text query into a 384-dimensional vector array
    query_vector = model.encode(vibe).tolist()
    
    # 2. Query the index collection for top 2 closest graph nodes based on Cosine Distance
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=2,
        include=["metadatas", "distances", "documents"]
    )
    
    formatted_results = []
    
    # Strictly validate that the DB returned data layers before trying to parse them
    if not results or not results.get('ids') or len(results['ids']) == 0 or len(results['ids'][0]) == 0:
        return formatted_results  # Safely return empty list to frontend if nothing matches

    # Safely iterate through valid found nodes
    for idx in range(len(results['ids'][0])):
        try:
            metadata = results['metadatas'][0][idx]
            raw_distance = results['distances'][0][idx]
            document_text = results['documents'][0][idx]
            
            # Convert raw cosine distance metric back into similarity percentages
            match_score = round(1.0 - raw_distance, 4)
            
            formatted_results.append(
                SearchResponse(
                    title=metadata.get('title', 'Unknown Title'),
                    artist=metadata.get('artist', 'Unknown Artist'),
                    match_score=max(0.0, match_score),  # Out-of-bounds protection
                    summary=document_text
                )
            )
        except IndexError:
            # Catch structural misalignments in empty array slices dynamically
            continue
            
    return formatted_results