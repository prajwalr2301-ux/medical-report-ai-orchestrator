"""
PDF Processing Utilities
Handles PDF text extraction and image processing
"""
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import io
import base64
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extract text from PDF using multiple methods for robustness.

    Args:
        pdf_path: Path to PDF file

    Returns:
        dict with:
            - full_text: Combined text from all pages
            - pages: List of text per page
            - page_count: Number of pages
            - method: Extraction method used
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Try PyMuPDF first (fast and reliable)
    try:
        return _extract_with_pymupdf(pdf_path)
    except Exception as e:
        print(f"⚠️  PyMuPDF failed: {e}, trying pdfplumber...")

    # Fallback to pdfplumber
    try:
        return _extract_with_pdfplumber(pdf_path)
    except Exception as e:
        raise Exception(f"All PDF extraction methods failed: {e}")


def _extract_with_pymupdf(pdf_path: Path) -> dict:
    """Extract text using PyMuPDF (fitz)"""
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        pages.append(text)

    doc.close()

    return {
        "full_text": "\n\n--- PAGE BREAK ---\n\n".join(pages),
        "pages": pages,
        "page_count": len(pages),
        "method": "PyMuPDF"
    }


def _extract_with_pdfplumber(pdf_path: Path) -> dict:
    """Extract text using pdfplumber (better for tables)"""
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            pages.append(text if text else "")

    return {
        "full_text": "\n\n--- PAGE BREAK ---\n\n".join(pages),
        "pages": pages,
        "page_count": len(pages),
        "method": "pdfplumber"
    }


def extract_images_from_pdf(pdf_path: str) -> list:
    """
    Extract images from PDF for potential OCR or analysis.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of dicts with image info:
            - page_num: Page number (0-indexed)
            - image_index: Image index on page
            - format: Image format (png, jpeg, etc.)
            - data: Base64 encoded image data
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)

            images.append({
                "page_num": page_num,
                "image_index": img_index,
                "format": base_image["ext"],
                "data": base64.b64encode(base_image["image"]).decode('utf-8')
            })

    doc.close()
    return images


def pdf_to_base64(pdf_path: str) -> str:
    """
    Convert PDF to base64 for sending to Gemini multimodal API.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Base64 encoded PDF data
    """
    pdf_path = Path(pdf_path)

    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()

    return base64.b64encode(pdf_bytes).decode('utf-8')


def create_sample_lab_report_text() -> str:
    """
    Creates a sample lab report text for testing when no PDF is available.
    This simulates extracted text from a typical lab report.
    """
    return """
CITY MEDICAL LABORATORY
123 Health Street, Medical City, ST 12345
Phone: (555) 123-4567

LABORATORY REPORT

Patient Information:
Name: John Doe
DOB: 01/15/1985
Gender: Male
Patient ID: PT-2024-001
Date of Collection: 01/15/2024
Date of Report: 01/16/2024

Ordering Physician: Dr. Sarah Smith, MD
Department: Internal Medicine

COMPLETE BLOOD COUNT (CBC)

Test Name                   Result      Unit        Reference Range         Flag
----------------------------------------------------------------------------------
Hemoglobin                  13.5        g/dL        13.0-17.0              NORMAL
Hematocrit                  40.2        %           38.0-50.0              NORMAL
Red Blood Cell Count        4.8         million/µL  4.5-5.5                NORMAL
White Blood Cell Count      8.2         thousand/µL 4.0-11.0               NORMAL
Platelet Count             250          thousand/µL 150-400                NORMAL

COMPREHENSIVE METABOLIC PANEL

Test Name                   Result      Unit        Reference Range         Flag
----------------------------------------------------------------------------------
Glucose (Fasting)           95          mg/dL       70-100                 NORMAL
Creatinine                  1.0         mg/dL       0.7-1.3                NORMAL
BUN                         15          mg/dL       7-20                   NORMAL
Sodium                      140         mmol/L      136-145                NORMAL
Potassium                   4.2         mmol/L      3.5-5.0                NORMAL
Calcium                     9.5         mg/dL       8.5-10.5               NORMAL

LIPID PANEL

Test Name                   Result      Unit        Reference Range         Flag
----------------------------------------------------------------------------------
Total Cholesterol           190         mg/dL       <200                   NORMAL
LDL Cholesterol             115         mg/dL       <100                   HIGH
HDL Cholesterol             55          mg/dL       >40                    NORMAL
Triglycerides              100          mg/dL       <150                   NORMAL

THYROID FUNCTION

Test Name                   Result      Unit        Reference Range         Flag
----------------------------------------------------------------------------------
TSH                         2.5         µIU/mL      0.5-5.0                NORMAL
Free T4                     1.2         ng/dL       0.8-1.8                NORMAL

Comments:
- LDL cholesterol slightly elevated. Recommend dietary modifications and follow-up in 3 months.
- All other values within normal limits.
- Patient is well-hydrated and fasting glucose indicates good metabolic control.

Report Authorized by: Dr. Emily Johnson, MD
Medical Laboratory Director
License: ML-12345

END OF REPORT
"""


if __name__ == "__main__":
    # Quick test
    print("PDF Utils Module Loaded")
    print("\nSample lab report text preview:")
    print(create_sample_lab_report_text()[:500] + "...")
