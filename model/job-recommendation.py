from functions import combine_files
import re
import pandas as pd
import string # for text cleaning
import contractions # for expanding short form words
import torch
from transformers import DistilBertTokenizer, DistilBertModel, BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np

def extract_details(resume_text):
    
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', resume_text)  # Remove non-ASCII characters
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple whitespaces with a single space
    cleaned_text = contractions.fix(cleaned_text)
    # Patterns to match skills and education sections
    skills_pattern = r'(?i)(skills|skill highlights)[:\n\r]*([\s\S]*?)(?=\n[A-Z]|education|$)'
    education_pattern = r'(?i)(education|education details)[:\n\r]*([\s\S]*?)(?=\n[A-Z]|skills|$)'
    
    # Extracting Skills
    skills_match = re.search(skills_pattern, cleaned_text, re.DOTALL)
    skills = skills_match.group(2).strip() if skills_match else None

    # Extracting Education
    education_match = re.search(education_pattern, cleaned_text, re.DOTALL)
    education = education_match.group(2).strip() if education_match else None
    
    return f"Skills: {skills} Education: {education}"

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

def distil_bert(details):
    # Initialize the DistilBERT tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    # Tokenize and embed job descriptions
    description_embeddings = []
    for description in details:
        tokens = tokenizer(description, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            output = model(**tokens)
        embeddings = output.last_hidden_state.mean(dim=1).numpy()
        description_embeddings.append(embeddings[0]) 
    return description_embeddings

def bert(details):
    # Initialize the DistilBERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    # Tokenize and embed job descriptions
    description_embeddings = []
    for description in details:
        tokens = tokenizer(description, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            output = model(**tokens)
        embeddings = output.last_hidden_state.mean(dim=1).numpy()
        description_embeddings.append(embeddings[0]) 
    return description_embeddings

def heuristic_match(job_skills,resume_skills):
    matching_skills = set(job_skills).intersection(set(resume_skills))
    if len(matching_skills) > 0:  
        return True
    return False

def extract_website(json_str):
    try:
        data_dict = json.loads(json_str)  # Convert JSON string to dictionary
        return data_dict.get("Website", None)  
    except Exception as e:
        return None 
    
def evaluate(similarity_scores):
    threshold = 0.7
    TP = FP = TN = FN = 0
    for i in range(len(similarity_scores)):  
        for j in range(len(similarity_scores[i])):  
            similarity = similarity_scores[i][j]
            job_title = job_data['skills'][i]
            resume_title = df_combined['Resume_str'][j]
            actual_match = heuristic_match(job_title,resume_title)
            if similarity > threshold:
                if actual_match:
                    TP+=1
                else:
                    FP+=1
            else:
                if actual_match:
                    FN+=1
                else:
                    TN+=1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)
    f1_score =  2 * ((precision * recall) / (precision + recall))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    return accuracy,precision,recall,f1_score



def compare(accuracy_bert, precision_bert, recall_bert, f1score_bert, accuracy_distlibert, precision_distilbert, recall_distilbert, f1score_distilbert):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    bert_scores = [accuracy_bert, precision_bert, recall_bert, f1score_bert]
    distilbert_scores = [accuracy_distlibert, precision_distilbert, recall_distilbert, f1score_distilbert]
    x = np.arange(len(metrics))  
    width = 0.35  

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_bert = ax.bar(x - width/2, bert_scores, width, label='BERT', color='#4CAF50', alpha=0.8)  
    bars_distilbert = ax.bar(x + width/2, distilbert_scores, width, label='DistilBERT', color='#FFC107', alpha=0.8) 

    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Model Comparison: Accuracy, Precision, Recall, F1 Score', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=10)  
    ax.set_ylim(0.92, 1.01) 
    ax.yaxis.grid(True, linestyle='--', alpha=0.6) 

    def add_values_to_bars(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    add_values_to_bars(bars_bert)
    add_values_to_bars(bars_distilbert)

    plt.tight_layout(pad=2)
    plt.savefig('model_comparison_fixed.png', dpi=300) 
    plt.show()


# Resume Reading + Extraction + Embeddings ##
filename1 = 'data/Resume.csv'
filename2 = 'data/UpdatedResumeDataSet.csv'
column_name1 = 'Resume_str' 
column_name2 = 'Category'  
df_combined = combine_files(filename1, filename2, column_name1, column_name2)
resume_details = df_combined['Resume_str'].apply(extract_details)
resume_embeddings = distil_bert(resume_details)
with open('resume_embeddings.pkl', 'wb') as f:
    pickle.dump(resume_embeddings, f)
print(" Resume Embeddings saved successfully!")
resume_embeddings_bert = bert(resume_details)
with open('resume_embeddings_bert.pkl', 'wb') as f:
    pickle.dump(resume_embeddings_bert, f)
print(" Resume Embeddings saved successfully!")

## Job Reading + Extraction + Embeddings ##
job_file = 'data/Jobs/job_descriptions.csv'
job_data = pd.read_csv(job_file)
job_details = job_data['Experience'] + ' '+ job_data['Qualifications'] + ' '+ job_data['Job Description']+ ' '+ job_data['Responsibilities'] + ' ' + job_data['skills']
job_details = job_details.apply(text_cleaning)
job_embeddings = distil_bert(job_details[:2000])
with open('job_embeddings.pkl', 'wb') as f:
    pickle.dump(job_embeddings, f)
print(" Job Embeddings saved successfully!")
job_embeddings_bert = bert(job_details[:2000])
with open('job_embeddings_bert.pkl', 'wb') as f:
    pickle.dump(job_embeddings_bert, f)
print(" Job Embeddings saved successfully!")


## Similarity ##
with open('resume_embeddings.pkl', 'rb') as f:
    resume_embeddings_distilbert = pickle.load(f)

with open('job_embeddings.pkl', 'rb') as f:
    job_embeddings_distilbert = pickle.load(f)

with open('resume_embeddings_bert.pkl', 'rb') as f:
    resume_embeddings_bert = pickle.load(f)

with open('job_embeddings_bert.pkl', 'rb') as f:
    job_embeddings_bert = pickle.load(f)

similarity_scores_bert = cosine_similarity(job_embeddings_bert, resume_embeddings_bert)
similarity_scores_distilbert = cosine_similarity(job_embeddings_distilbert,resume_embeddings_distilbert)

# Top 5 Job Recommendation ##
num_top_jobs = 5
top_jobs = []
job_data['Website'] = job_data['Company Profile'].apply(extract_website)
# Loop through each resume's similarity scores and rank the jobs
for i, resume_similarity in enumerate(similarity_scores_distilbert):
    jobs_with_scores = list(enumerate(resume_similarity))
    jobs_with_scores.sort(key=lambda x: x[1], reverse=True)
    top_jobs_for_resume = jobs_with_scores[:num_top_jobs]
    top_jobs.append(top_jobs_for_resume)

for i, resume_similarity in enumerate(top_jobs):
    print(f"Top jobs for Resume {i+1} - Category: {df_combined['Category'][i]}")  
    for job_index, score in resume_similarity:
        job_title = job_data['Job Title'][job_index]
        company_name = job_data['Company'][job_index]
        website_link = job_data['Website'][job_index]
        print(f"  Job {job_index + 1} - Similarity Score: {score:.4f} - {job_title} at {company_name} - Website: {website_link}")


## Model Evaluation ##
accuracy_bert,precision_bert,recall_bert,f1score_bert = evaluate(similarity_scores_bert)
accuracy_distlibert,precision_distilbert,recall_distilbert,f1score_distilbert = evaluate(similarity_scores_distilbert)
compare(accuracy_bert, precision_bert, recall_bert, f1score_bert, accuracy_distlibert, precision_distilbert, recall_distilbert, f1score_distilbert)