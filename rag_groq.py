import os
import io
import json
import hashlib
import re
import shutil
import tempfile
import pickle
from typing import List, Dict, Any

import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st

# -------------------------
# CONFIG & INITIALIZATION
# -------------------------

# 1. Dynamic Tesseract Pathing
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# For Linux/Mac/Streamlit Cloud, Tesseract is expected to be in the system PATH

# 2. Secure API Key Handling
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
USE_GROQ = bool(GROQ_API_KEY)

if USE_GROQ:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        MODEL_ID = "llama-3.1-8b-instant"
    except Exception:
        groq_client = None
        USE_GROQ = False
else:
    groq_client = None

_EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CACHE_DIR = ".rag_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------------
# UTILITIES
# -------------------------
def get_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -------------------------
# CORE RAG LOGIC
# -------------------------
class FaissStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas: List[Dict[str, Any]] = []

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadatas.extend(metadatas)

    def search(self, q_emb: np.ndarray, k: int = 5):
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)
        results = []
        for id_ in I[0]:
            if id_ != -1 and id_ < len(self.metadatas):
                results.append(self.metadatas[id_])
        return results

@st.cache_resource
def get_embedder():
    return SentenceTransformer(_EMBED_MODEL_NAME)

def extract_content(pdf_path: str, output_dir: str) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    pages = []
    
    # Text/Table extraction
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            tables = page.extract_tables() or []
            pages.append({'page_number': i+1, 'text': text, 'tables': tables})

    # Visual extraction (OCR + Images)
    doc = fitz.open(pdf_path)
    images = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        
        # OCR the page to catch text in images
        try:
            ocr_text = pytesseract.image_to_string(img).strip()
        except:
            ocr_text = ""
        
        existing_text = pages[i]['text']
        pages[i]['text'] = f"{existing_text}\n\n{ocr_text}".strip()

        # Save individual images for UI display
        for img_idx, imginfo in enumerate(page.get_images(full=True)):
            xref = imginfo[0]
            base_image = doc.extract_image(xref)
            try:
                img_obj = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
                img_name = f"p{i+1}_img{img_idx}.png"
                img_path = os.path.join(output_dir, img_name)
                img_obj.save(img_path)
                images.append({'page': i+1, 'path': img_path})
            except:
                continue
    
    return {'pages': pages, 'images': images}

def chunk_text(text: str, max_words: int = 200, overlap: int = 40) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
        if i + max_words >= len(words): break
    return chunks

def process_pdf(pdf_path: str, f_hash: str):
    index_path = os.path.join(CACHE_DIR, f"{f_hash}.faiss")
    meta_path = os.path.join(CACHE_DIR, f"{f_hash}.meta.pkl")
    struct_path = os.path.join(CACHE_DIR, f"{f_hash}.struct.json")

    # Load from cache if exists
    if all(os.path.exists(p) for p in [index_path, meta_path, struct_path]):
        with open(meta_path, "rb") as f: metadatas = pickle.load(f)
        with open(struct_path, "r") as f: doc_struct = json.load(f)
        index = faiss.read_index(index_path)
        store = FaissStore(index.d)
        store.index, store.metadatas = index, metadatas
        return store, doc_struct

    # Otherwise, build
    extract_dir = os.path.join(CACHE_DIR, f_hash + "_assets")
    doc_struct = extract_content(pdf_path, extract_dir)
    
    chunks, chunk_metas = [], []
    for p in doc_struct['pages']:
        if p['text']:
            for c in chunk_text(p['text']):
                chunks.append(c)
                chunk_metas.append({'text': c, 'page_number': p['page_number']})

    embedder = get_embedder()
    embeddings = embedder.encode(chunks, convert_to_numpy=True).astype("float32")
    store = FaissStore(embeddings.shape[1])
    store.add(embeddings, chunk_metas)

    # Save cache
    faiss.write_index(store.index, index_path)
    with open(meta_path, "wb") as f: pickle.dump(chunk_metas, f)
    save_json(struct_path, doc_struct)
    
    return store, doc_struct

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Pro RAG — Groq", layout="wide")

st.title("📄 Multi-Modal RAG QA")
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    file_bytes = uploaded_file.read()
    f_hash = get_file_hash(file_bytes)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    with st.spinner("Analyzing document..."):
        store, doc_struct = process_pdf(tmp_path, f_hash)
    
    os.remove(tmp_path) # Clean up temp

    # Sidebar Info
    st.sidebar.success("Document Indexed")
    st.sidebar.info(f"Pages: {len(doc_struct['pages'])} | Images: {len(doc_struct['images'])}")

    # Main UI
    col1, col2 = columns = st.columns([1, 1])
    
    with col1:
        st.subheader("Query Document")
        query = st.text_input("Ask anything:")
        
        if query:
            q_emb = get_embedder().encode([query], convert_to_numpy=True).astype("float32")
            results = store.search(q_emb, k=5)
            
            context = "\n\n".join([f"[Page {r['page_number']}]: {r['text']}" for r in results])
            
            if USE_GROQ:
                prompt = f"Use context to answer. Question: {query}\n\nContext: {context}"
                res = groq_client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[{"role": "user", "content": prompt}]
                )
                st.markdown(f"### Answer\n{res.choices[0].message.content}")
            else:
                st.warning("Groq key missing. Showing top snippet:")
                st.info(results[0]['text'])

            with st.expander("View Source Passages"):
                for r in results:
                    st.write(f"**Page {r['page_number']}**: {r['text'][:300]}...")

    with col2:
        st.subheader("Extracted Visuals")
        if doc_struct['images']:
            for img in doc_struct['images'][:6]: # Show first 6
                st.image(img['path'], caption=f"Source: Page {img['page']}")
        else:
            st.write("No images detected.")