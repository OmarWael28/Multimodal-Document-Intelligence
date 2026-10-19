# Multimodal Document Intelligence

A simple local AI application that analyzes images and PDFs using Gemma 4 through Ollama and extracts structured information.

---

## 📌 Project Overview

This application accepts document images (`.jpg`, `.jpeg`, `.png`) or PDFs (`.pdf`), classifies them (invoice, receipt, unknown), and extracts key financial details into a consistent, validated JSON structure. It operates completely **offline and free of charge** using a local multimodal model (`gemma4:e4b`) served via Ollama.

---

## ✨ Features

- **Multi-Format Input**: Supports `.jpg`, `.jpeg`, `.png`, and `.pdf` files.
- **Document Classification**: Automatically classifies documents into `invoice`, `receipt`, or `unknown`.
- **Information Extraction**: Extracts company name, invoice number, date, total amount, and currency.
- **Missing Field Handling**: Sets missing fields explicitly to `"value": null` rather than hallucinating.
- **Uncertain Field Detection**: Flags unclear, low-quality, or blurred text with `"uncertain": true`.
- **Consistent Pydantic Schema**: Guarantees predictable JSON response structure.
- **100% Local & Free**: No paid APIs, cloud services, vector databases, or complex frameworks.

---

## 🛠️ Technology Stack

- **Language**: Python 3.12+
- **API Framework**: FastAPI & Uvicorn
- **Data Validation**: Pydantic v2
- **Local AI Engine**: Ollama (`gemma4:e4b`)
- **PDF Processing**: PyMuPDF (`pymupdf`)
- **Image Processing**: Pillow (`PIL`)

---

## 📁 Project Structure

```text
document-intelligence/
│
├── main.py                # FastAPI application and Ollama integration
├── requirements.txt       # Minimal required Python dependencies
├── README.md              # Project documentation and setup guide
└── samples/               # Test sample documents
    ├── invoice1.jpg       # Complete clear invoice
    ├── invoice2.jpg       # Invoice with missing invoice number
    ├── receipt1.jpg       # Store receipt
    ├── invoice_unclear.jpg# Document with blurred amount
    ├── unknown.jpg        # Non-document image (landscape)
    └── invoice_sample.pdf # Sample PDF document
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
Ensure [Ollama](https://ollama.com) is installed and running on your system. Verify that `gemma4:e4b` is installed:

```bash
ollama list
```

You should see:
```text
NAME         ID              SIZE      MODIFIED
gemma4:e4b   c6eb396dbd59    9.6 GB    ...
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI Application

```bash
uvicorn main:app --reload
```

The server will start at `http://127.0.0.1:8000`.

### 4. Interactive API Documentation

Open your browser and navigate to:
```text
http://127.0.0.1:8000/docs
```

Use the Swagger UI to test the `POST /analyze` endpoint by uploading any file from the `samples/` directory.

---

## 💡 Example API Output

### Clear Invoice Output (`POST /analyze` with `invoice1.jpg`)

```json
{
  "document_type": "invoice",
  "company_name": {
    "value": "TechCorp Solutions Ltd.",
    "uncertain": false
  },
  "invoice_number": {
    "value": "INV-2026-001",
    "uncertain": false
  },
  "date": {
    "value": "2026-10-05",
    "uncertain": false
  },
  "amount": {
    "value": "1500.00",
    "uncertain": false
  },
  "currency": {
    "value": "USD",
    "uncertain": false
  },
  "status": "valid"
}
```

### Missing Field Handling (`invoice2.jpg`)

```json
{
  "document_type": "invoice",
  "company_name": {
    "value": "Acme Global Services",
    "uncertain": false
  },
  "invoice_number": {
    "value": null,
    "uncertain": false
  },
  "date": {
    "value": "2026-09-15",
    "uncertain": false
  },
  "amount": {
    "value": "500.00",
    "uncertain": false
  },
  "currency": {
    "value": "EGP",
    "uncertain": false
  },
  "status": "missing_invoice_number"
}
```

---

## 📋 Internship Requirement Mapping

| Requirement | Implementation Details |
|---|---|
| **File/Image Input** | Upload endpoint supports `.jpg`, `.jpeg`, `.png`, and `.pdf` |
| **Information Extraction** | Multimodal analysis using local `gemma4:e4b` via Ollama |
| **Classification** | Classifies document type as `invoice`, `receipt`, or `unknown` |
| **Structured Output** | Enforced consistent JSON output using Pydantic `DocumentResult` |
| **Missing Fields** | Explicit `value: null` returned when data is absent |
| **Uncertain Fields** | Flags blurry or low-quality data with `uncertain: true` |
| **PDF Handling** | Converts first page of PDF to image using PyMuPDF |
| **Validation** | Error responses returned for empty, corrupted, or unsupported files |
| **Sample Files** | Ready-to-use test documents included in `samples/` directory |

---

## ⚠️ Limitations & Notes

- **Local Model Constraints**: Uses `gemma4:e4b` locally. Processing speed depends on system GPU/CPU performance.
- **PDF Page Limit**: Processes the **first page** of uploaded PDF documents.
- **OCR Quality**: Extraction accuracy depends on the visual clarity, resolution, and orientation of the uploaded file.
- **Simple Architecture**: Designed as a lean, lightweight demonstration app without complex OCR pipelines or external paid APIs.

---

## 📹 Demo Video Script (2-3 Minutes)

1. **Introduction**: Show `ollama list` and start server with `uvicorn main:app --reload`.
2. **Demo 1 (Standard Invoice)**: Upload `samples/invoice1.jpg` in `/docs`. Show full extracted JSON.
3. **Demo 2 (Receipt)**: Upload `samples/receipt1.jpg`. Show classification as `receipt`.
4. **Demo 3 (Missing Field)**: Upload `samples/invoice2.jpg`. Point out `"invoice_number": {"value": null}`.
5. **Demo 4 (Unclear Field)**: Upload `samples/invoice_unclear.jpg`. Point out `"uncertain": true`.
6. **Demo 5 (PDF & Unsupported File)**: Upload `samples/invoice_sample.pdf` and an invalid file to show validation.
