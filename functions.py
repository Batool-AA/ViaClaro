import tkinter as tk
from tkinter import filedialog
import PyPDF2
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.utils import shuffle
import openai

le = LabelEncoder()
tfidf = TfidfVectorizer(stop_words='english')

def select_pdf_file():
    root = tk.Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    return file_path

def pdf_to_string(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)  
    cleanText = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]', ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

def creating_vectors(filename,column):
    df = pd.read_csv(filename)
    df_shuffled = shuffle(df, random_state=42)
    df = df_shuffled
    df[column] = df[column].apply(lambda x: cleanResume(x))
    tfidf.fit(df[column])
    requiredText  = tfidf.transform(df[column])
    return df, requiredText, tfidf

def assigning_categories(df,column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    unique_values = df[column].unique()
    return le, unique_values

openai.api_key = ''

def generate_roadmap(domain):
    prompt = f"Create a comprehensive roadmap for ${domain}. The roadmap should contain all the key skills to excel in this profession. Also mention the dependencies between these skills."
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-4" if you have access
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    roadmap = response['choices'][0]['message']['content']
    return roadmap