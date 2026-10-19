# app/extractor.py
# ─────────────────────────────────────────────────────────────
# PIPELINE STEP 1: Extract text from uploaded ARAS documents
#
# INPUT:  raw file bytes + filename  (e.g. b"..." + "ItemType_Guide.pdf")
# OUTPUT: plain text string          (e.g. "To create an ItemType...")
#
# Supported formats: PDF, DOCX, TXT
# Each format needs a different library to read it.
# ─────────────────────────────────────────────────────────────

import io  # io.BytesIO turns raw bytes into a file-like object


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Read a file and return its text content as a plain string.

    Args:
        file_bytes: The raw binary content of the uploaded file
        filename:   The original filename (used to detect the file type)

    Returns:
        A single string with all the text from the document
    """

    # Get the file extension and convert to lowercase
    # e.g. "Workflow_Config.PDF" → "pdf"
    extension = filename.split(".")[-1].lower()

    if extension == "pdf":
        return _extract_from_pdf(file_bytes)

    elif extension == "docx":
        return _extract_from_docx(file_bytes)

    elif extension == "txt":
        return _extract_from_txt(file_bytes)

    else:
        # Raise a clear error so the API can return a helpful message
        raise ValueError(f"Unsupported file type: '{extension}'. Use PDF, DOCX, or TXT.")


def _extract_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF file.
    We use the 'pypdf' library — it reads each page and returns its text.
    """
    import pypdf

    # Wrap bytes in a file-like object so pypdf can read it
    pdf_file = io.BytesIO(file_bytes)
    reader = pypdf.PdfReader(pdf_file)

    # Loop through every page and collect the text
    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # Some pages might be images/diagrams with no text
            pages_text.append(page_text)

    # Join all pages with a newline between them
    return "\n".join(pages_text)


def _extract_from_docx(file_bytes: bytes) -> str:
    """
    Extract text from a Word document (.docx).
    We use the 'python-docx' library — it reads paragraphs one by one.
    ARAS development guides are often shared as Word docs with numbered steps.
    """
    import docx

    # Wrap bytes in a file-like object
    docx_file = io.BytesIO(file_bytes)
    document = docx.Document(docx_file)

    # Each paragraph is a separate block of text in Word
    paragraphs = [para.text for para in document.paragraphs if para.text.strip()]

    return "\n".join(paragraphs)


def _extract_from_txt(file_bytes: bytes) -> str:
    """
    Extract text from a plain text file.
    Just decode the bytes — no special library needed.
    """
    # 'errors="replace"' means if there's a weird character, use ? instead of crashing
    return file_bytes.decode("utf-8", errors="replace")
