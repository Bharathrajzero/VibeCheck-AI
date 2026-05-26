# VibeCheck AI // Semantic Vector-Space Audio Discovery Engine

VibeCheck AI is a full-stack semantic audio discovery engine that maps human emotions, abstract scenarios, and descriptive queries directly to curated audio tracks using high-dimensional vector similarity. Instead of keyword filtering, it leverages embeddings and vector search to deliver results that truly match the *vibe*.

---

## 🛠️ Tech Stack

- **Backend:** FastAPI (Python, async web framework)
- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`, 384-dim vectors)
- **Vector Database:** ChromaDB (HNSW graph-based indices, persistent storage)
- **Similarity Metric:** Cosine Distance
- **Frontend:** HTML5, TailwindCSS, Async JavaScript Fetch

---

## 🚀 System Flow

```
User Query ("driving through neon city at 2 AM")
   │
   ▼
FastAPI Gateway (/api/search)
   │
Sentence-Transformer → 384-dim vector
   │
ChromaDB → nearest neighbor search
   │
Cosine Distance → similarity %
   │
Frontend → glowing UI cards (<15ms latency)
```

---

## 📂 Project Structure

```text
vibecheck/
│
├── data/
│   └── songs.csv              # Dataset with 100+ descriptive entries
├── database/                  # Auto-generated HNSW index files
├── static/
│   └── index.html             # Vantablack + Radium Green frontend
├── main.py                    # FastAPI server core
├── ingest.py                  # Vector DB ingestion utility
└── requirements.txt           # Dependencies
```

---

## ⚡ Deployment Guide

### 1. Setup Environment
```bash
git clone https://github.com/your-username/vibecheck-ai.git
cd vibecheck-ai
pip install -r requirements.txt
```

### 2. Ingest Data
```bash
python ingest.py
```

### 3. Run Server
```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Access
Open: **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

---

## 🎯 Example Queries

- `spiritual peace sufi journey` → maps to *Kun Faya Kun*  
- `heavy metal rock screaming heartbreak` → isolates *Bekhayali*, *Sadda Haq*  
- `cozy acoustic evening bedroom pop` → targets Anuv Jain’s *Baarishein*, *Kasoor*  

---

## 🛡️ Monitoring & Safety

- Swagger Docs: `http://127.0.0.1:8000/docs`  
- Health Route: `/health` for DB + model checks  
- Exception Handling: clean empty states instead of crashes  

---

## 🤝 Contributing

1. Fork the repo  
2. Create a feature branch (`git checkout -b feature-name`)  
3. Commit changes (`git commit -m "Add feature"`)  
4. Push branch (`git push origin feature-name`)  
5. Open a Pull Request  

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.
```

---

This version is **ready to paste into your repo root** as `README.md`. It’s concise, professional, and GitHub‑friendly.  

Would you like me to also prepare a **`requirements.txt` file** with all dependencies pinned to stable versions, so anyone cloning your repo can install everything in one go?
