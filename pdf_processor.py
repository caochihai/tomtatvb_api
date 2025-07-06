import pdfplumber
import re
from fuzzywuzzy import fuzz
import os
import unicodedata
import mimetypes
import requests
from typing import Tuple, Optional
from io import BytesIO
import tempfile
import gdown

class PDFProcessor:
    """A class to process and clean text from PDF files, with a focus on Vietnamese text.
    Supports both local file paths and URLs, including Google Drive and GitHub."""
    
    # Class constants
    VIETNAMESE_PATTERN = r'[àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]'
    PHRASES_TO_REMOVE = [
        "cộng hòa xã hội chủ nghĩa việt nam",
        "độc lập - tự do - hạnh phúc"
    ]
    SIMILARITY_THRESHOLD = 90
    SUPPORTED_URL_SCHEMES = ['http', 'https']

    def __init__(self, file_path_or_url: str):
        """Initialize PDFProcessor with a PDF file path or URL.
        
        Args:
            file_path_or_url (str): Path to a local PDF file or URL to a PDF file.
        """
        self._file_path_or_url = file_path_or_url
        self._extracted_text: Optional[str] = None
        self._is_url = self._is_valid_url(file_path_or_url)

    def _is_valid_url(self, path_or_url: str) -> bool:
        """Check if the input is a valid URL with supported scheme.
        
        Args:
            path_or_url (str): The input string to check.
            
        Returns:
            bool: True if the input is a valid URL, False otherwise.
        """
        try:
            result = re.match(r'^(https?://)', path_or_url, re.IGNORECASE)
            return bool(result and result.group(1).lower().startswith(tuple(self.SUPPORTED_URL_SCHEMES)))
        except re.error:
            return False

    def _validate_file(self) -> bool:
        """Validate if the input is a valid local PDF file or accessible URL.
        
        Returns:
            bool: True if the input is valid, False otherwise.
        """
        if self._is_url:
            try:
                response = requests.head(self._file_path_or_url, allow_redirects=True, timeout=5)
                return response.status_code == 200
            except requests.RequestException:
                return False
        else:
            mime_type, _ = mimetypes.guess_type(self._file_path_or_url)
            return (os.path.isfile(self._file_path_or_url) and 
                    self._file_path_or_url.lower().endswith('.pdf') and 
                    mime_type == 'application/pdf')

    def _extract_text_from_pdf(self) -> None:
        """Extract text from all pages of the PDF file, handling both local files and URLs."""
        self._extracted_text = ""
        try:
            if self._is_url and 'drive.google.com' in self._file_path_or_url:
                # Handle Google Drive URLs using gdown
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                try:
                    gdown.download(self._file_path_or_url, temp_file.name, quiet=False)
                    with pdfplumber.open(temp_file.name) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                self._extracted_text += page_text + " "
                finally:
                    os.unlink(temp_file.name)  # Delete temporary file
            else:
                # Handle other URLs (GitHub, Dropbox, etc.) or local files
                if self._is_url:
                    response = requests.get(self._file_path_or_url, stream=True, timeout=10)
                    response.raise_for_status()
                    pdf_file = BytesIO(response.content)
                else:
                    pdf_file = self._file_path_or_url
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            self._extracted_text += page_text + " "
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")

    def _has_vietnamese_text(self) -> bool:
        """Check if the extracted text contains Vietnamese characters.
        
        Returns:
            bool: True if Vietnamese characters are found, False otherwise.
        """
        if not self._extracted_text:
            return False
        return bool(re.search(self.VIETNAMESE_PATTERN, self._extracted_text))

    def _normalize_text(self, text: str) -> str:
        """Normalize text to NFC form and clean up whitespace.
        
        Args:
            text (str): Input text to normalize.
            
        Returns:
            str: Normalized text with cleaned whitespace.
        """
        normalized = unicodedata.normalize('NFC', text)
        return re.sub(r'\s+', ' ', normalized).strip()

    def _remove_unwanted_phrases(self) -> None:
        """Remove specified phrases from the extracted text using fuzzy matching."""
        if not self._extracted_text:
            return

        lines = self._extracted_text.split('\n')
        processed_lines = []
        
        for line in lines:
            processed_line = line
            normalized_line = self._normalize_text(line).lower()
            
            for phrase in self.PHRASES_TO_REMOVE:
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                matches = pattern.finditer(normalized_line)
                
                for match in matches:
                    matched_text = match.group()
                    if fuzz.partial_ratio(matched_text.lower(), phrase) >= self.SIMILARITY_THRESHOLD:
                        processed_line = re.sub(
                            re.escape(matched_text), 
                            '', 
                            processed_line, 
                            flags=re.IGNORECASE
                        )
            processed_lines.append(processed_line.strip())
            
        self._extracted_text = '\n'.join(processed_lines)

    def _clean_text(self) -> None:
        """Clean the extracted text by normalizing, removing phrases, and cleaning whitespace."""
        if not self._extracted_text:
            return
            
        self._extracted_text = unicodedata.normalize('NFC', self._extracted_text)
        self._extracted_text = self._extracted_text.replace('\r', '')
        self._remove_unwanted_phrases()
        self._extracted_text = re.sub(r'\s+', ' ', self._extracted_text).strip()

    def process_pdf(self) -> Tuple[bool, bool]:
        """Process the PDF file (local or URL) and extract cleaned text.
        
        Returns:
            Tuple[bool, bool]: 
                - First bool: True if file/URL is valid and processed, False otherwise
                - Second bool: True if Vietnamese text is detected, False otherwise
        """
        if not self._validate_file():
            return False, False
            
        try:
            self._extract_text_from_pdf()
            has_vietnamese = self._has_vietnamese_text()
            if has_vietnamese:
                self._clean_text()
            return True, has_vietnamese
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return False, False

    def get_extracted_text(self) -> Optional[str]:
        """Get the extracted and cleaned text from the PDF.
        
        Returns:
            Optional[str]: The extracted text, or None if not processed.
        """
        return self._extracted_text