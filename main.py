import io
import json
import os
from typing import Any, Literal, Optional

import pymupdf
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
import pytesseract
from pydantic import BaseModel, ConfigDict, ValidationError
from fastapi.responses import PlainTextResponse

app = FastAPI(
    title="Multimodal Document Intelligence API",
    description="Analyze invoices, receipts, and documents using Gemma 4 via Ollama.",
    version="1.1.0",
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_PIXELS = 25_000_000

TESSERACT_CMD = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
ALLOWED_DOCUMENT_TYPES = {"invoice", "receipt", "unknown"}


class FieldResult(BaseModel):
    value: Optional[str] = None
    uncertain: bool = False


class DocumentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: Literal["invoice", "receipt", "unknown"]
    company_name: FieldResult
    invoice_number: FieldResult
    date: FieldResult
    amount: FieldResult
    currency: FieldResult
    status: str


EXTRACTION_PROMPT = """
You are a document information extraction assistant.

Analyze the supplied document text (OCR) and extract. The image may also be provided as supporting evidence:

- document_type: invoice, receipt, or unknown
- company_name
- invoice_number
- date
- amount: total amount, not an individual item amount
- currency

Rules:
1. Only extract information visible in the document.
2. Never invent missing values.
3. Missing fields must have value null and uncertain false.
4. Present but unreadable fields must have value null and uncertain true.
5. If a value is only partially readable, return it only if useful
   and mark uncertain true.
6. Classify unrelated images as unknown.
7. For receipts, invoice_number may be null if no receipt number exists.
8. Treat OCR text as the primary source. Do not invent details from the image.
9. Return only JSON using exactly this structure:

{
  "document_type": "invoice",
  "company_name": {"value": "ABC Company", "uncertain": false},
  "invoice_number": {"value": "INV-1001", "uncertain": false},
  "date": {"value": "2026-10-05", "uncertain": false},
  "amount": {"value": "1500.00", "uncertain": false},
  "currency": {"value": "USD", "uncertain": false},
  "status": "valid"
}

The status field is only a placeholder. The application will
calculate the final status independently.
"""


def normalize_field(field_data: Any) -> FieldResult:
    """Convert an LLM field into a consistent, validated structure."""

    if field_data is None:
        return FieldResult()

    if isinstance(field_data, dict):
        value = field_data.get("value")
        uncertain = field_data.get("uncertain", False)

        if not isinstance(uncertain, bool):
            raise ValueError("Field uncertain must be a boolean.")

    else:
        value = field_data
        uncertain = False

    if value is not None and not isinstance(value, (str, int, float)):
        raise ValueError("Invalid field value.")

    if value is None:
        return FieldResult(value=None, uncertain=uncertain)

    value = str(value).strip()

    if value.lower() in {"", "null", "none", "n/a", "not provided"}:
        return FieldResult(value=None, uncertain=uncertain)

    return FieldResult(value=value, uncertain=uncertain)


def calculate_status(result: DocumentResult) -> str:
    """Determine status from validated extracted fields."""

    if result.document_type == "unknown":
        return "unknown_document"

    fields = [
        result.company_name,
        result.date,
        result.amount,
        result.currency,
    ]

    if result.document_type == "invoice":
        if result.invoice_number.value is None:
            if result.invoice_number.uncertain:
                return "uncertain_fields"
            return "missing_invoice_number"

        fields.append(result.invoice_number)

    # Uncertainty takes priority over missing information.
    if any(field.uncertain for field in fields):
        return "uncertain_fields"

    if any(field.value is None for field in fields):
        return "missing_fields"

    return "valid"


def prepare_image(contents: bytes, extension: str) -> tuple[bytes, str, str]:
    """Validate a file, render PDFs when needed, and extract text with OCR."""

    embedded_pdf_text = ""

    if extension == ".pdf":
        try:
            with pymupdf.open(stream=contents, filetype="pdf") as pdf:
                if pdf.needs_pass:
                    raise HTTPException(
                        status_code=400,
                        detail="Password-protected PDFs are not supported.",
                    )

                if len(pdf) == 0:
                    raise HTTPException(
                        status_code=400,
                        detail="The PDF has no pages.",
                    )

                # Intentionally process only the first page.
                page = pdf[0]
                embedded_pdf_text = page.get_text() or ""
                pixmap = page.get_pixmap(dpi=200, alpha=False)
                image_bytes = pixmap.tobytes("png")

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted PDF file.",
            ) from exc
    else:
        try:
            with Image.open(io.BytesIO(contents)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise HTTPException(
                        status_code=400,
                        detail="File content does not match a supported image format.",
                    )

                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(
                        status_code=413,
                        detail="Image dimensions are too large.",
                    )

                image.load()
                image = image.convert("RGB")

                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()

        except HTTPException:
            raise
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(
                status_code=400,
                detail="The uploaded image is invalid or corrupted.",
            ) from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as ocr_image:
            ocr_text = pytesseract.image_to_string(ocr_image, config="--psm 6")
    except pytesseract.TesseractNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Tesseract OCR was not found. Install it or set the "
                "TESSERACT_CMD environment variable."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from the uploaded document.",
        ) from exc

    # Prefer embedded PDF text, while keeping OCR as a fallback.
    if embedded_pdf_text.strip():
        combined_text = (
            "PDF embedded text:\n"
            + embedded_pdf_text[:20_000]
            + "\n\nOCR text:\n"
            + ocr_text[:20_000]
        )
    else:
        combined_text = ocr_text[:20_000]

    return image_bytes, combined_text, ocr_text


@app.get("/", response_class=PlainTextResponse)
def home():
    return "Multimodal Document Intelligence API"


@app.post("/analyze", response_model=DocumentResult)
async def analyze_document(file: UploadFile = File(...)):
    filename = file.filename or ""
    extension = os.path.splitext(filename)[1].lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload JPG, JPEG, PNG, or PDF.",
        )

    # Read at most one byte beyond the allowed size.
    try:
        contents = await file.read(MAX_FILE_SIZE + 1)
    finally:
        await file.close()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 10 MB limit.",
        )

    image_bytes, extracted_text, ocr_text = prepare_image(contents, extension)

    if not extracted_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No readable text was found in the document by OCR.",
        )

    prompt = EXTRACTION_PROMPT + (
        "\n\nOCR-extracted document text (primary evidence):\n"
        + extracted_text[:20_000]
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }

    try:
        response = requests.post(
            f"{OLLAMA_URL.rstrip('/')}/api/chat",
            json=payload,
            timeout=180,
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail="Ollama could not process the document. Check the model and Ollama logs.",
            )

        response_data = response.json()
        llm_content = response_data.get("message", {}).get("content", "")

        if not isinstance(llm_content, str) or not llm_content.strip():
            raise ValueError("The model returned an empty response.")

        parsed = json.loads(llm_content)

        if not isinstance(parsed, dict):
            raise ValueError("The model response must be a JSON object.")

        doc_type = parsed.get("document_type")

        if not isinstance(doc_type, str):
            raise ValueError("Missing or invalid document type.")

        doc_type = doc_type.strip().lower()

        if doc_type not in ALLOWED_DOCUMENT_TYPES:
            raise ValueError("Unsupported document type.")

        result = DocumentResult(
            document_type=doc_type,
            company_name=normalize_field(parsed.get("company_name")),
            invoice_number=normalize_field(parsed.get("invoice_number")),
            date=normalize_field(parsed.get("date")),
            amount=normalize_field(parsed.get("amount")),
            currency=normalize_field(parsed.get("currency")),
            status="pending_validation",
        )

        # Never trust the model to decide whether a document is valid.
        result.status = calculate_status(result)

        return result

    except HTTPException:
        raise
    except requests.exceptions.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail="Ollama is unavailable. Make sure Ollama is running.",
        ) from exc
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail="The document could not be processed into valid structured data.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the document.",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
