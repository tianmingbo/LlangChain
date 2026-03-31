from pypdf import PdfWriter

out_path = "rag/loader/assets/sample.pdf"

writer = PdfWriter()
writer.add_blank_page(width=595, height=842)
writer.add_metadata(
    {
        "/Title": "PDF Loader Demo",
        "/Author": "LlangChain",
        "/Subject": "Sample PDF for loader test",
    }
)

with open(out_path, "wb") as f:
    writer.write(f)

print(out_path)
