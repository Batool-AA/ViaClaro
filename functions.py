import tkinter as tk
from tkinter import filedialog
import PyPDF2
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import openai

def select_pdf_file():
    root = tk.Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    return file_path

import PyPDF2
import pytesseract
from pdf2image import convert_from_path
import io

def pdf_to_string(file):
    # Ensure that the file is being read correctly in Flask
    try:
        # Read the file as a binary stream using file.read()
        file_stream = io.BytesIO(file.read())

        # Try to extract text from the PDF
        reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        # If no text is found (possibly an image-based PDF), use OCR
        if not text.strip():
            print("No text found in PDF, trying OCR...")
            images = convert_from_path(file_stream)
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image) + "\n"

        return text
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return str(e)


def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)  
    cleanText = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]', ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

def combine_files(file1, file2, column1, column2):
    df1 = pd.read_csv(file1)  
    df2 = pd.read_csv(file2) 
    df1[column1] = df1[column1].fillna('').apply(lambda x: cleanResume(x) if isinstance(x, str) else '')
    df2[column1] = df2[column1].fillna('').apply(lambda x: cleanResume(x) if isinstance(x, str) else '')
    df_combined = pd.concat([df1[[column1, column2]], df2[[column1, column2]]], ignore_index=True)
    return df_combined

def creating_vectors(df_combined, column1):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit(df_combined[column1])
    requiredText = tfidf.transform(df_combined[column1])
    return requiredText, tfidf

def assigning_categories(df,column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    unique_values = df[column].unique()
    return le, unique_values

def generate_roadmap(domain):
    openai.api_key = 'sk-SyD8VVSdgBAhB1wcAjdEKSDW5xf6S9ufTrBiE1sXPmT3BlbkFJA6q6fxFrGZKt7ByzU6SYnaxESShyOhk5B9LWFCZWYA'
    if (openai.api_key != ''):
        prompt = f"Create a comprehensive roadmap for ${domain}. The roadmap should contain all the key skills to excel in this profession."
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        roadmap = response['choices'][0]['message']['content']
        return roadmap
    else:
        return "No API Key."