"""Text extraction service for txt, pdf, and docx files."""

import io
import logging
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class TextExtractor:
    """Extracts plain text from supported document formats."""

    def extract(self, content: bytes, filename: str, content_type: str) -> str:
        """
        Extract text from raw document bytes.

        Dispatches to the appropriate parser based on MIME type, falling back
        to file extension when the MIME type is ``application/octet-stream``.

        Raises:
            ValueError: If the format is not supported.
            ImportError: If the required optional parser library is not installed.
        """
        extractor = self._resolve_extractor(content_type, filename)
        logger.info(
            "Extracting text from %r (content_type=%r)", filename, content_type
        )
        return extractor(content)

    def _resolve_extractor(
        self, content_type: str, filename: str
    ) -> Callable[[bytes], str]:
        by_mime: dict[str, Callable[[bytes], str]] = {
            "text/plain": self._extract_txt,
            "application/pdf": self._extract_pdf,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": self._extract_docx,
        }

        if content_type in by_mime:
            return by_mime[content_type]

        by_ext: dict[str, Callable[[bytes], str]] = {
            ".txt": self._extract_txt,
            ".pdf": self._extract_pdf,
            ".docx": self._extract_docx,
        }
        ext = Path(filename).suffix.lower()
        if ext in by_ext:
            return by_ext[ext]

        raise ValueError(
            f"Unsupported document format: content_type={content_type!r}, "
            f"extension={ext!r}. Supported formats: .txt, .pdf, .docx"
        )

    def _extract_txt(self, content: bytes) -> str:
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Unable to decode text file: no supported encoding matched.")

    def _extract_pdf(self, content: bytes) -> str:
        import PyPDF2  # noqa: PLC0415

        reader = PyPDF2.PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _extract_docx(self, content: bytes) -> str:
        import docx  # noqa: PLC0415

        doc = docx.Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
