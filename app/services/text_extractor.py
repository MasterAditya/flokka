"""Text extraction service for txt, pdf, and docx files."""

import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TextExtractor:
    """Extracts plain text from supported document formats."""

    SUPPORTED_TYPES = {
        "text/plain": "_extract_txt",
        "application/pdf": "_extract_pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "_extract_docx",
        "application/octet-stream": "_extract_by_extension",
    }

    def extract(self, content: bytes, filename: str, content_type: str) -> str:
        """
        Extract text from document bytes.

        Args:
            content: Raw file bytes.
            filename: Original filename (used for extension fallback).
            content_type: MIME type of the file.

        Returns:
            Extracted plain text string.

        Raises:
            ValueError: If the file type is unsupported.
        """
        method_name = self.SUPPORTED_TYPES.get(content_type)
        if method_name == "_extract_by_extension":
            method_name = self._resolve_by_extension(filename)

        if method_name is None:
            raise ValueError(
                f"Unsupported content type: {content_type!r}. "
                f"Supported types: {', '.join(self.SUPPORTED_TYPES)}"
            )

        method = getattr(self, method_name)
        logger.info("Extracting text from %r using %s", filename, method_name)
        return method(content, filename)

    # ------------------------------------------------------------------
    # Private extraction methods
    # ------------------------------------------------------------------

    def _resolve_by_extension(self, filename: str) -> str:
        """Map file extension to extraction method name."""
        ext = Path(filename).suffix.lower()
        mapping = {
            ".txt": "_extract_txt",
            ".pdf": "_extract_pdf",
            ".docx": "_extract_docx",
        }
        method_name = mapping.get(ext)
        if method_name is None:
            raise ValueError(
                f"Unsupported file extension: {ext!r}. Supported: .txt, .pdf, .docx"
            )
        return method_name

    def _extract_txt(self, content: bytes, filename: str) -> str:  # noqa: ARG002
        """Decode plain-text file."""
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Unable to decode text file with any supported encoding.")

    def _extract_pdf(self, content: bytes, filename: str) -> str:  # noqa: ARG002
        """Extract text from PDF using PyPDF2."""
        try:
            import PyPDF2  # noqa: PLC0415

            reader = PyPDF2.PdfReader(io.BytesIO(content))
            pages: list[str] = []
            for page in reader.pages:
                text = page.extract_text() or ""
                pages.append(text)
            return "\n".join(pages)
        except ImportError as exc:
            raise ImportError("PyPDF2 is required for PDF extraction.") from exc

    def _extract_docx(self, content: bytes, filename: str) -> str:  # noqa: ARG002
        """Extract text from DOCX using python-docx."""
        try:
            import docx  # noqa: PLC0415

            doc = docx.Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError as exc:
            raise ImportError("python-docx is required for DOCX extraction.") from exc
