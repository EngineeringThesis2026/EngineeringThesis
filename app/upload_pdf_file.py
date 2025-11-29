import streamlit as st
from pypdf import PdfReader


def extract_text_from_pdf(file):
    pdf_text = ""
    st.write("File uploaded successfully!")
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    return pdf_text
