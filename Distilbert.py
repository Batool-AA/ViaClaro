import os
import re
import pdfplumber
import pandas as pd
import numpy as np
import string 
import contractions 
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
from functions import select_pdf_file
import pickle
import json

# Define a function to extract information from a PDF
def extract_information(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        resume_text = ""
        for page in pdf.pages:
            resume_text = " ".join([resume_text, page.extract_text()])
    resume_text = resume_text.strip()
    return resume_text


def extract_details(resume_text):
    # Define regular expressions to extract Skills & Education
    skills_pattern = r'Skills\n([\s\S]*?)(?=\n[A-Z]|$)' 
    education_pattern = r'Education\n([\s\S]*?)(?=\n[A-Z][a-z]*\n|$)'
    
    skills_match = re.findall(skills_pattern, resume_text, re.DOTALL)
    education_match = re.findall(education_pattern, resume_text, re.DOTALL)
    
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


## Resume  Extraction + Embeddings ##
pdf = select_pdf_file()
resume_text = extract_information(pdf)
skills_education = extract_details(resume_text)
resume_cleaned = skills_education['Skills'] + ' ' + skills_education['Education']
resume_cleaned = text_cleaning(resume_cleaned)




tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

## Jobs Loading + Embeddings ##
job_file = 'data/Jobs/job_descriptions.csv'
job_data = pd.read_csv(job_file)
job_data= job_data[:2000]
with open('job_embeddings.pkl', 'rb') as f:
    job_embeddings_distilbert = pickle.load(f)

#Tokenize and embed resume
resume_embeddings = []
tokens = tokenizer(resume_cleaned, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    output = model(**tokens)
embeddings = output.last_hidden_state.mean(dim=1).numpy()
resume_embeddings.append(embeddings[0])  

## Similarity + Recommendation ##
similarity_scores = cosine_similarity(job_embeddings_distilbert, resume_embeddings)
top_jobs = 5
resume_topjobs =[]
job_similarity_scores = []
for j in range(len(job_data)):
    score = similarity_scores[j][0]  
    job_similarity_scores.append((j, score))
job_similarity_scores.sort(key=lambda x: x[1], reverse=True)
resume_topjobs=job_similarity_scores[:top_jobs]
print("Top 5 jobs for your resume:")
job_data['Website'] = job_data['Company Profile'].apply(extract_website)
for job_index, score in resume_topjobs:
     job_title = job_data['Job Title'][job_index]
     company_name = job_data['Company'][job_index]
     website_link = job_data['Website'][job_index]
     print(f"  Job {job_index + 1} - Similarity Score: {score:.4f} - {job_title} at {company_name} - Website: {website_link}")

