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

def creating_vectors(file1, file2, column1, column2):
    df1 = pd.read_csv(file1)  # First file with Resume and Category
    df2 = pd.read_csv(file2)  # Second file with Resume and Category
    # Clean resumes in both datasets
    df1[column1] = df1[column1].fillna('').apply(lambda x: cleanResume(x) if isinstance(x, str) else '')
    df2[column1] = df2[column1].fillna('').apply(lambda x: cleanResume(x) if isinstance(x, str) else '')
    # Concatenate the dataframes along rows, ensuring categories and resumes are included
    df_combined = pd.concat([df1[[column1, column2]], df2[[column1, column2]]], ignore_index=True)
    # Fit TF-IDF on the combined resume text
    tfidf.fit(df_combined[column1])
    requiredText = tfidf.transform(df_combined[column1])
    return df_combined, requiredText, tfidf


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