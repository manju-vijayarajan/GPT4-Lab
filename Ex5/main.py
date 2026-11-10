import gradio as gr
import requests
from PyPDF2 import PdfReader
from docx import Document
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"
def extract_text(file):
    if file is None:
        return "No file uploaded."
    filename = file.name
    text = ""
    try:
        if filename.lower().endswith(".pdf"):
            reader = PdfReader(filename)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        elif filename.lower().endswith(".docx"):
            doc = Document(filename)
            text = "\n".join(
                paragraph.text
                for paragraph in doc.paragraphs
                if paragraph.text.strip()
            )
        elif filename.lower().endswith(".txt"):
            with open(filename,"r",encoding="utf-8") as f:
                text = f.read()
        else:
            return "Unsupported file type."
        return text.strip()
    except Exception as e:
        return f"Error extracting document: {e}"
def ollama_generate(prompt):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "temperature": 0.2,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response","No response generated.").strip()
    except requests.exceptions.ConnectionError:
        return (
            "Error: Could not connect to Ollama. "
            "Make sure Ollama is running."
        )
    except requests.exceptions.Timeout:
        return "Error: Ollama request timed out."
    except Exception as e:
        return f"Error: {e}"
def analyze_document(file, question):
    text = extract_text(file)
    if text.startswith("Error") or text == "Unsupported file type.":
        return text
    if not text:
        return "No readable text found in the document."
    summary_prompt = f"""
You are an AI document analysis assistant.

Analyze the following document and provide a clear
and concise summary.

Document:
{text}

Summary:
"""
    summary = ollama_generate(summary_prompt)
    result = f"--- SUMMARY ---\n\n{summary}"
    if question and question.strip():
        qa_prompt = f"""
You are an AI document analysis assistant.

Answer the user's question using only the
information available in the document.

Document:
{text}

Question:
{question}

Answer:
"""
        answer = ollama_generate(qa_prompt)
        result += (
            f"\n\n--- QUESTION & ANSWER ---\n\n"
            f"Question: {question}\n\n"
            f"Answer: {answer}"
        )
    return result
interface = gr.Interface(
    fn=analyze_document,
    inputs=[
        gr.File(file_types=[".pdf", ".docx",".txt"],label="Upload Document"),
        gr.Textbox(lines=3,placeholder=(
                "Ask a question about the "
                "uploaded document..."
            ),label="Ask Question")
    ],
    outputs=gr.Textbox(lines=25,label="Analysis Result"),
    title="AI Document Analyzer",
    description=(
        "Upload a PDF, DOCX, or TXT document "
        "to generate a summary and ask questions "
        "about its content."
    )
)
if __name__ == "__main__":
    interface.launch()