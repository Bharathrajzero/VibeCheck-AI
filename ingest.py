import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import os

def ingest_data():
    print("🔄 Step 1: Initializing SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded successfully.")
    
    print("🔄 Step 2: Connecting to persistent ChromaDB client...")
    chroma_client = chromadb.PersistentClient(path="./database")
    collection = chroma_client.get_or_create_collection(name="vibe_tracks")
    
    if not os.path.exists("data/songs.csv"):
        print("❌ Error: data/songs.csv not found! Please create the file.")
        return

    print("🔄 Step 3: Reading data/songs.csv...")
    df = pd.read_csv("data/songs.csv")
    
    documents = []
    embeddings = []
    metadatas = []
    ids = []
    
    print(f"🔄 Step 4: Encoding vectors for {len(df)} tracks...")
    for idx, row in df.iterrows():
        combined_text = f"{row['title']} by {row['artist']}: {row['description']}"
        vector = model.encode(row['description']).tolist()
        
        documents.append(combined_text)
        embeddings.append(vector)
        metadatas.append({"title": row['title'], "artist": row['artist']})
        ids.append(str(idx))
        print(f"  🔹 Vectorized: {row['title']} by {row['artist']}")
        
    print("🔄 Step 5: Writing vector arrays into HNSW index space...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )
    print("🚀 SUCCESS: Ingestion complete! Data mapped into vector storage.")

if __name__ == "__main__":
    ingest_data()