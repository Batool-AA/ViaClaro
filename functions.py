import tkinter as tk
from tkinter import filedialog
import PyPDF2
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import openai
import string 
import contractions 
import json
import io
import pdfplumber

def extract_information(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        resume_text = ""
        for page in pdf.pages:
            resume_text = " ".join([resume_text, page.extract_text()])
    resume_text = resume_text.strip()
    return resume_text

def select_pdf_file():
    root = tk.Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    return file_path



def pdf_to_string(file):
    # Ensure that the file is being read correctly in Flask
    try:
        # Read the file as a binary stream using file.read()
        file_stream = io.BytesIO(file.read())

        # Try to extract text from the PDF
        reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in reader.pages:
            text = " ".join([text, page.extract_text()])
        text = text.strip()
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
        prompt = f"Give a brief description of what {domain} is and create a comprehensive roadmap for {domain}. The roadmap should contain all the key skills to excel in this profession."
        
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
    
def extract_details(resume_text):
    # print("in extract:", resume_text)
    # Define regular expressions to extract Skills & Education
    skills_pattern = r'Skills\n([\s\S]*?)(?=\n[A-Z]|$)' 
    education_pattern = r'Education\n([\s\S]*?)(?=\n[A-Z][a-z]*\n|$)'
    
    skills_match = re.findall(skills_pattern, resume_text, re.DOTALL)
    education_match = re.findall(education_pattern, resume_text, re.DOTALL)
    # print("s:", skills_match, '\n', "e:", education_match, '\n')
    if len(skills_match)!=0:
        skills = skills_match[0]
    else:
        skills_pattern = r'skills\n((?:.*)*)' 
        skills_match = re.findall(skills_pattern, resume_text, re.DOTALL)
        if len(skills_match)!=0:
            skills = skills_match[0]
        else:
            skills = None
            
    if len(education_match)!=0:
        education = education_match[0]
    else:
        education = None
    
    return {
        'Skills': skills,
        'Education': education
    }

def text_cleaning(text:str) -> str:
    if pd.isnull(text):
        return
    text = text.lower().strip()
    translator = str.maketrans('', '', string.punctuation)
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text) # Remove URLs
    text = re.sub(r'\S+@\S+', '', text) # Remove emails
    text = re.sub(r'\b\d{1,3}[-./]?\d{1,3}[-./]?\d{1,4}\b', '', text) # Remove phone numbers
    text = text.translate(translator) # Remove puctuations
    text = re.sub(r'[^a-zA-Z]', ' ', text) # Remove other non-alphanumeric characters
    
    return text.strip()

def extract_website(json_str):
    try:
        data_dict = json.loads(json_str)  # Convert JSON string to dictionary
        return data_dict.get("Website", None) 
    except Exception as e:
        return None 