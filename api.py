from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, constr
from typing import Dict, Any
import logging
from cachetools import TTLCache
from pdf_processor import PDFProcessor  # Import from pdf_processor.py
from text_summarizer import TextSummarizer  # Import from text_summarizer.py
import os

# Configure logging for server deployment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),  # Save logs to file
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# Cache for processed PDFs (100 items, 1-hour TTL)
cache = TTLCache(maxsize=100, ttl=3600)

# Initialize FastAPI app
app = FastAPI(
    title="PDF Summarization API",
    description="API for summarizing PDF documents with Vietnamese text support.",
    version="1.0.0"
)

# Add CORS middleware with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cấp quyền truy cập api * là all
    allow_credentials=True,
    allow_methods=["POST"], # Cấp quyền method
    allow_headers=["*"],
)

# Initialize TextSummarizer
try:
    summarizer = TextSummarizer()
except Exception as e:
    logger.error(f"Failed to initialize TextSummarizer: {str(e)}")
    raise RuntimeError(f"Failed to initialize summarizer: {str(e)}")

class PDFRequest(BaseModel):
    """Pydantic model for PDF summarization request."""
    pdf_path: constr(max_length=2048)  # Limit path length

def create_error_response(detail: str, error_type: str = "ProcessingError") -> Dict[str, Any]:
    return {
        "error": error_type,
        "detail": detail
    }

@app.post("/summarize_pdf", response_model=Dict[str, Any])
async def summarize_pdf(request: PDFRequest = Body(...)):
    """Summarize text extracted from a PDF file (local path or URL)."""
    pdf_path = request.pdf_path.strip()
    logger.info(f"Processing PDF request for: {pdf_path}")

    if not pdf_path:
        logger.error("Empty pdf_path provided")
        raise HTTPException(
            status_code=400,
            detail=create_error_response("pdf_path cannot be empty", "InvalidRequest")
        )

    # Check cache
    if pdf_path in cache:
        logger.info(f"Using cached summary for: {pdf_path}")
        return JSONResponse(content={"summary": cache[pdf_path]})

    try:
        # Initialize and process PDF
        pdf_processor = PDFProcessor(pdf_path)
        success, has_vietnamese = pdf_processor.process_pdf()

        if not success:
            logger.error(f"Failed to process PDF: {pdf_path}")
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    f"PDF file '{pdf_path}' does not exist or is inaccessible.",
                    "FileError"
                )
            )

        if not has_vietnamese:
            logger.warning(f"PDF does not contain Vietnamese text: {pdf_path}")
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "PDF may be a scanned image or in an unsupported language.",
                    "ContentError"
                )
            )

        extracted_text = pdf_processor.get_extracted_text()
        if not extracted_text:
            logger.error(f"No text extracted from PDF: {pdf_path}")
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "No text could be extracted from the PDF.",
                    "ContentError"
                )
            )

        # Summarize text
        summary = summarizer.summarize(extracted_text)
        if not summary or summary.startswith("Error"):
            logger.error(f"Summarization failed for PDF: {pdf_path}")
            raise HTTPException(
                status_code=500,
                detail=create_error_response(
                    summary if summary.startswith("Error") else "Summarization failed.",
                    "SummarizationError"
                )
            )

        # Cache the summary
        cache[pdf_path] = summary
        logger.info(f"Successfully summarized and cached PDF: {pdf_path}")
        return JSONResponse(
            content={"summary": summary},
            media_type="application/json; charset=utf-8"
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error processing PDF '{pdf_path}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                f"An unexpected error occurred: {str(e)}",
                "InternalServerError"
            )
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=5000,
        log_level="info",
        workers=os.cpu_count() or 4
    )
