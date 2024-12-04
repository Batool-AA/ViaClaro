import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
from functions import select_pdf_file, extract_website, extract_details, text_cleaning, extract_information
import pickle

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
job_data = pd.read_csv(job_file, nrows=2000)
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

