"""
Text extraction utilities for preprocessing uploaded files.
Supports: plain text (.txt, .md), PDF (.pdf), and Word documents (.docx only).
"""

import io
from typing import Tuple


def extract_text_from_txt(content: bytes) -> Tuple[str, str]:
    """
    Extract text from plain text file bytes.
    Tries UTF-8 first, falls back to Latin-1.
    
    Returns:
        Tuple of (extracted_text, error_message). 
        If successful, error_message is empty string.
    """
    try:
        text = content.decode("utf-8")
        return text, ""
    except UnicodeDecodeError:
        pass
    
    try:
        text = content.decode("latin-1")
        return text, ""
    except Exception as e:
        return "", f"Failed to decode text file: {str(e)}"


def extract_text_from_pdf(content: bytes) -> Tuple[str, str]:
    """
    Extract text from PDF file bytes using pdfplumber.
    
    Returns:
        Tuple of (extracted_text, error_message).
        If successful, error_message is empty string.
    """
    try:
        import pdfplumber
    except ImportError:
        return "", "pdfplumber is not installed. Run: pip install pdfplumber"
    
    try:
        pdf_file = io.BytesIO(content)
        text_parts = []
        
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        full_text = "\n".join(text_parts)
        return full_text, ""
    except Exception as e:
        return "", f"Failed to extract text from PDF: {str(e)}"


def extract_text_from_docx(content: bytes) -> Tuple[str, str]:
    """
    Extract text from Word .docx file bytes using python-docx.
    Note: Only .docx files are supported, not legacy .doc files.
    
    Returns:
        Tuple of (extracted_text, error_message).
        If successful, error_message is empty string.
    """
    try:
        from docx import Document
    except ImportError:
        return "", "python-docx is not installed. Run: pip install python-docx"
    
    try:
        docx_file = io.BytesIO(content)
        doc = Document(docx_file)
        
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        full_text = "\n".join(paragraphs)
        return full_text, ""
    except Exception as e:
        return "", f"Failed to extract text from DOCX: {str(e)}"
