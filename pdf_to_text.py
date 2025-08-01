import pdfplumber

pdf_path = "data/Original Sources/_OceanofPDF.com_Tony.pdf"
txt_path = "data/Tony.txt"

with pdfplumber.open(pdf_path) as pdf:
    all_text = ""
    for page in pdf.pages:
        all_text += page.extract_text() + "\n"

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(all_text)

print(f"Extracted text saved to {txt_path}")