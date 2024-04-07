import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import docx
from PyPDF2 import PdfReader
import tempfile
import os


model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def summarize_text(text):
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def read_text(file_path):
    file_type = file_path.split(".")[-1]
    if file_type == "txt":
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    elif file_type == "docx":
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    elif file_type == "pdf":
        pdf_reader = PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    else:
        
        text = ""
    return text

def main():
    st.title("SummarifyAI 📝")
    
    text = st.text_area("Enter text: ")
    if st.button("Summarize"):
        summary = summarize_text(text)
        st.write("Summary: ")
        st.write(summary)

    uploaded_file = st.file_uploader("Upload a file", type=["txt", "docx", "pdf"])

    if uploaded_file is not None:
        file_contents = uploaded_file.getvalue()

        if uploaded_file.type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            file_text = read_text(temp_file_path)
            os.unlink(temp_file_path)  
        else:
            file_text = file_contents

        

        text = read_text(uploaded_file.name)
        summary = summarize_text(text)
        
        st.write("Summary:")
        st.write(summary)

if __name__ == "__main__":
    main()