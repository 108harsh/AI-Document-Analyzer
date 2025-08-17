import os
import io
import json
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from PyPDF2 import PdfReader
import chardet  # For encoding detection
from pdf2image import convert_from_bytes
import pytesseract

# ----------------------------------------------------
# Logging & Environment
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env file")
genai.configure(api_key=GEMINI_KEY)

# ----------------------------------------------------
# FastAPI Setup & CORS
# ----------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Change to your frontend origin if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Static Files Setup
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
    logger.warning(f"Static folder created at {STATIC_DIR}. Place your app.html here.")
# Commented out to avoid decode errors from StaticFiles
# app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ----------------------------------------------------
# Database Setup (SQLite with SQLAlchemy)
# ----------------------------------------------------
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DB_PATH = os.path.join(BASE_DIR, "appdata.db")
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class PDFDocument(Base):
    __tablename__ = "pdf_documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    content = Column(LargeBinary, nullable=False)
    extracted_text = Column(Text, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

class QueryRecord(Base):
    __tablename__ = "query_records"
    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    response_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# In-memory document store for RAG (for now, keep as list of extracted_text)
DOCUMENTS = []

# ----------------------------------------------------
# Request Model
# ----------------------------------------------------
class QueryRequest(BaseModel):
    query: str

# ----------------------------------------------------
# Global Exception Handler
# ----------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    tb_str = traceback.format_exc()
    logger.error(f"Unhandled exception: {exc}\nTraceback:\n{tb_str}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": tb_str}
    )

# ----------------------------------------------------
# Gemini RAG Function
# ----------------------------------------------------
def get_rag_response(query: str) -> dict:
    if not DOCUMENTS:
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")

    # --- Keyword-based chunk selection ---
    # Split documents into paragraphs (chunks)
    all_paragraphs = []
    for doc in DOCUMENTS:
        # Split on double newlines or single newlines for paragraphs
        paragraphs = [p.strip() for p in doc.split('\n\n') if p.strip()]
        if len(paragraphs) < 5:
            # If not enough, split on single newlines
            paragraphs = [p.strip() for p in doc.split('\n') if p.strip()]
        all_paragraphs.extend(paragraphs)

    # Extract keywords from query (simple split, lowercase, remove stopwords)
    import re
    stopwords = set(["the", "is", "at", "which", "on", "and", "a", "an", "of", "to", "in", "for", "by", "with", "as", "that", "this", "it", "from", "or", "be", "are", "was", "were", "has", "have", "had", "but", "not", "no", "do", "does", "did"])
    query_words = set(re.findall(r"\w+", query.lower())) - stopwords
    if not query_words:
        query_words = set(re.findall(r"\w+", query.lower()))


    # Filter out address-like and very short paragraphs
    def is_content_paragraph(para):
        # Must be at least 100 chars and contain a period (full sentence)
        if len(para) < 100:
            return False
        if '.' not in para:
            return False
        # Exclude paragraphs with mostly address-like keywords
        address_keywords = ["floor", "road", "palace", "block", "main road", "street", "avenue", "sector", "building", "pincode", "pin code", "no.", "plot", "lane", "village", "district", "taluk", "ward", "post", "city", "state", "zip", "address"]
        para_lower = para.lower()
        address_count = sum(1 for k in address_keywords if k in para_lower)
        # If more than 2 address keywords, likely an address
        if address_count > 2:
            return False
        return True

    # Score paragraphs by keyword overlap, but only if content-rich
    scored_paragraphs = []
    for para in all_paragraphs:
        if not is_content_paragraph(para):
            continue
        para_words = set(re.findall(r"\w+", para.lower()))
        overlap = len(query_words & para_words)
        if overlap > 0:
            scored_paragraphs.append((overlap, para))

    # Sort by overlap, then by length (descending)
    scored_paragraphs.sort(key=lambda x: (-x[0], -len(x[1])))

    # Select top N relevant paragraphs, up to ~4000 chars
    selected = []
    total_chars = 0
    max_chars = 4000
    for _, para in scored_paragraphs:
        if total_chars + len(para) > max_chars:
            break
        selected.append(para)
        total_chars += len(para)
    # Fallback: if nothing matched, use the top 3 longest content-rich paragraphs
    if not selected:
        # Find all content-rich paragraphs
        content_paragraphs = [p for p in all_paragraphs if is_content_paragraph(p)]
        # Sort by length descending
        content_paragraphs.sort(key=lambda p: -len(p))
        selected = content_paragraphs[:3] if content_paragraphs else ["\n\n".join(DOCUMENTS)[:2000]]

    context_text = "\n\n".join(selected)
    prompt_text = (
        "Answer the following question using the context provided.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        'Respond strictly in this JSON format: '
        '{"decision": "APPROVED or REJECTED or UNKNOWN", "justification": "<short justification>", "source_content": ["<at least 2-3 relevant, content-rich sentences from the context used for answer>"]}'
        ' If you do not know the answer, set decision to "UNKNOWN", justification to "No justification provided", and source_content to an empty list.'
        ' Do not include any other text or explanation. Do not include markdown or code block formatting.'
    )

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt_text)
    if not response or not getattr(response, "text", None):
        raise HTTPException(status_code=500, detail="No response from Gemini API.")

    raw_text = response.text
    logger.info(f"Raw Gemini response: {raw_text}")

    # Remove code block wrappers (e.g., ```json ... ```)
    cleaned = raw_text.strip()
    if cleaned.startswith('```'):
        cleaned = re.sub(r'^```[a-zA-Z]*', '', cleaned)
        cleaned = cleaned.strip('`\n')

    # Try to extract the first valid JSON object from the response
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            result = json.loads(json_str)
            if not isinstance(result, dict):
                result = {}
            decision = result.get("decision", "UNKNOWN")
            justification = result.get("justification", "No justification provided")
            source_content = result.get("source_content", [])
            if not isinstance(source_content, list):
                source_content = []
            return {
                "decision": decision,
                "justification": justification,
                "source_content": source_content
            }
        except Exception as e:
            logger.warning(f"Failed to parse JSON from Gemini response: {e}")

    # Fallback: if no JSON found or parse failed
    return {
        "decision": "UNKNOWN",
        "justification": "No justification provided",
        "source_content": []
    }

# ----------------------------------------------------
# Routes
# ----------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend HTML page."""
    index_file = os.path.join(STATIC_DIR, "app.html")
    if not os.path.exists(index_file):
        return HTMLResponse("<h2>No app.html found in static/ folder.</h2>", status_code=200)
    # Read file as bytes, detect encoding, and fallback if needed
    try:
        with open(index_file, "rb") as f:
            content_bytes = f.read()
        import chardet
        detected = chardet.detect(content_bytes)
        encoding = detected.get("encoding")
        logger.info(f"Detected encoding for app.html: {encoding}")
        text = None
        decode_errors = []
        if encoding:
            try:
                text = content_bytes.decode(encoding, errors="strict")
                logger.info(f"Decoded app.html with detected encoding: {encoding}")
            except Exception as decode_err:
                decode_errors.append(f"Detected encoding '{encoding}': {decode_err}")
        if text is None:
            try:
                text = content_bytes.decode("utf-8", errors="replace")
                logger.info(f"Decoded app.html with utf-8 (replace)")
            except Exception as utf8_err:
                decode_errors.append(f"utf-8: {utf8_err}")
        if text is None:
            try:
                text = content_bytes.decode("latin-1", errors="replace")
                logger.info(f"Decoded app.html with latin-1 (replace)")
            except Exception as latin1_err:
                decode_errors.append(f"latin-1: {latin1_err}")
        if text is None:
            logger.error(f"All decoding attempts failed for app.html: {decode_errors}")
            return HTMLResponse("<h2>Could not decode app.html.</h2>", status_code=500)
        return HTMLResponse(text)
    except Exception as e:
        logger.exception(f"Error reading app.html: {e}")
        return HTMLResponse(f"<h2>Error reading app.html: {e}</h2>", status_code=500)


@app.post("/upload-document/")
async def upload_document(file: UploadFile):
    """Upload PDF or text, store in DB, and keep extracted text for RAG."""
    db = SessionLocal()
    try:
        content = await file.read()
        ext = file.filename.split('.')[-1].lower()
        text = ""

        if ext == "pdf":
            pdf_reader = PdfReader(io.BytesIO(content))
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
            # OCR fallback if no text extracted
            if not text.strip():
                logger.info(f"No text extracted with PyPDF2, trying OCR fallback for {file.filename}")
                try:
                    images = convert_from_bytes(content)
                    ocr_text = ""
                    for i, image in enumerate(images):
                        page_text = pytesseract.image_to_string(image)
                        ocr_text += page_text
                    text = ocr_text
                    logger.info(f"OCR extracted {len(text)} chars from {file.filename}")
                except Exception as ocr_err:
                    logger.error(f"OCR extraction failed for {file.filename}: {ocr_err}")
                    text = ""

            # Table extraction using camelot
            try:
                import tempfile
                import camelot
                import os
                tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                try:
                    tmp_pdf.write(content)
                    tmp_pdf.flush()
                    tmp_pdf.close()
                    tables = camelot.read_pdf(tmp_pdf.name, pages='all')
                    table_texts = []
                    for i, table in enumerate(tables):
                        table_str = table.df.to_string(index=False, header=True)
                        table_texts.append(f"[Table {i+1}]\n" + table_str)
                    if table_texts:
                        text += '\n\n' + '\n\n'.join(table_texts)
                        logger.info(f"Extracted {len(table_texts)} tables from {file.filename}")
                finally:
                    try:
                        os.unlink(tmp_pdf.name)
                    except Exception as cleanup_err:
                        logger.warning(f"Could not delete temp file {tmp_pdf.name}: {cleanup_err}")
            except Exception as table_err:
                logger.warning(f"Table extraction failed for {file.filename}: {table_err}")
        else:
            # Detect encoding for non-PDF files
            detected = chardet.detect(content)
            encoding = detected.get("encoding")
            logger.info(f"Detected encoding for {file.filename}: {encoding}")
            decode_errors = []
            # Try detected encoding first
            if encoding:
                try:
                    text = content.decode(encoding, errors="strict")
                    logger.info(f"Decoded {file.filename} with detected encoding: {encoding}")
                except Exception as decode_err:
                    decode_errors.append(f"Detected encoding '{encoding}': {decode_err}")
            # Try utf-8 with replace
            if not text:
                try:
                    text = content.decode("utf-8", errors="replace")
                    logger.info(f"Decoded {file.filename} with utf-8 (replace)")
                except Exception as utf8_err:
                    decode_errors.append(f"utf-8: {utf8_err}")
            # Try latin-1 with replace
            if not text:
                try:
                    text = content.decode("latin-1", errors="replace")
                    logger.info(f"Decoded {file.filename} with latin-1 (replace)")
                except Exception as latin1_err:
                    decode_errors.append(f"latin-1: {latin1_err}")
            if not text:
                logger.error(f"All decoding attempts failed for {file.filename}: {decode_errors}")
                raise HTTPException(status_code=400, detail=f"Could not decode file: {file.filename}")

        if not text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in file.")

        logger.info(f"Extracted text (first 500 chars): {text[:500]}")
        DOCUMENTS.append(text)
        # Store in DB
        pdf_doc = PDFDocument(filename=file.filename, content=content, extracted_text=text)
        db.add(pdf_doc)
        db.commit()
        logger.info(f"Stored document '{file.filename}', {len(text)} chars in DB.")
        return {"message": f"{file.filename} uploaded successfully", "total_docs": len(DOCUMENTS)}
    except Exception as e:
        logger.exception(f"Error processing file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

from fastapi import Form


@app.post("/query-document/")
async def query_document(query: str = Form(...)):
    """Query the uploaded documents, store query and response in DB, and return JSON answer."""
    db = SessionLocal()
    try:
        response = get_rag_response(query)
        # Store query and response in DB
        record = QueryRecord(query=query, response_json=json.dumps(response))
        db.add(record)
        db.commit()
        return response
    finally:
        db.close()

# ----------------------------------------------------
# Data Viewing Endpoints
# ----------------------------------------------------
from fastapi.responses import StreamingResponse

@app.get("/list-documents/")
def list_documents():
    """List all uploaded documents (metadata only)."""
    db = SessionLocal()
    try:
        docs = db.query(PDFDocument).all()
        return [
            {"id": d.id, "filename": d.filename, "uploaded_at": d.uploaded_at.isoformat(), "size": len(d.content)}
            for d in docs
        ]
    finally:
        db.close()

@app.get("/download-document/{doc_id}")
def download_document(doc_id: int):
    """Download a PDF document by ID."""
    db = SessionLocal()
    try:
        doc = db.query(PDFDocument).filter(PDFDocument.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return StreamingResponse(io.BytesIO(doc.content), media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename={doc.filename}"})
    finally:
        db.close()

@app.get("/list-queries/")
def list_queries():
    """List all queries and their responses."""
    db = SessionLocal()
    try:
        queries = db.query(QueryRecord).order_by(QueryRecord.created_at.desc()).all()
        return [
            {"id": q.id, "query": q.query, "response": json.loads(q.response_json), "created_at": q.created_at.isoformat()}
            for q in queries
        ]
    finally:
        db.close()

# ----------------------------------------------------
# Local Run
# ----------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))   # Railway gives PORT, fallback=8000
    uvicorn.run("backend:app", host="0.0.0.0", port=port, reload=True)

