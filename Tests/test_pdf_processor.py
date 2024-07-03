import unittest
from PyPDF2 import PdfReader
from src.pdf.pdf_processor import PDFProcessor

class TestPDFProcessor(unittest.TestCase):
    def setUp(self):
        self.pdf_file = 'sample.pdf'
        with open(self.pdf_file, 'wb') as f:
            f.write(b'%PDF-1.4\n%...')  # Minimal valid PDF content
        self.processor = PDFProcessor(self.pdf_file)

    def test_process_pdf(self):
        chunks = self.processor.process_pdf()
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)

if __name__ == '__main__':
    unittest.main()
