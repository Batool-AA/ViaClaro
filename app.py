from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from functions import pdf_to_string, cleanResume, generate_roadmap
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from functions import extract_details, text_cleaning, extract_website, extract_information
import torch

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS) for all routes

# Load models and other necessary files
try:
    with open('tfidf.pkl', 'rb') as tfidf_file:
        tfidf = pickle.load(tfidf_file)
    with open('ann_clf.pkl', 'rb') as ann_file:
        ann_clf = pickle.load(ann_file)
    with open('label_encoder.pkl', 'rb') as le_file:
        le_loaded = pickle.load(le_file)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    ## Jobs Loading + Embeddings ##
    job_file = 'data/Jobs/job_descriptions.csv'
    job_data = pd.read_csv(job_file, nrows=2000)
    with open('job_embeddings.pkl', 'rb') as f:
        job_embeddings_distilbert = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Route to handle the resume upload and return career paths
@app.route('/api/career', methods=['POST'])
def upload_resume():
    try:
        # Ensure a file is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Check if the file is a valid PDF
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Invalid file format, only PDF files are allowed'}), 400
        
        # Process the resume
        myresume = extract_information(file)
        cleaned_resume = cleanResume(myresume)
        input_features = tfidf.transform([cleaned_resume])
        predicted_probabilities = ann_clf.predict_proba(input_features)

        # Get top 5 career paths
        top_n = 5
        top_n_indices = np.argsort(predicted_probabilities[0])[-top_n:][::-1]
        top_categories = le_loaded.inverse_transform(top_n_indices)
        top_probabilities = predicted_probabilities[0][top_n_indices]

        career_paths = [
            {'category': category, 'probability': f"{probability:.4f}"}
            for category, probability in zip(top_categories, top_probabilities)
        ]

        return jsonify({'topCareerPaths': career_paths})

    except Exception as e:
        print(f"Error during file upload processing: {e}")
        return jsonify({'error': str(e)}), 500

# Route to return the roadmap for a specific career path
@app.route('/api/roadmap', methods=['POST'])
def get_roadmap():
    try:
        career_path = request.json.get('careerPath')
        if not career_path:
            return jsonify({'error': 'No career path selected'}), 400
        
        roadmap = generate_roadmap(career_path)
        return jsonify({'roadmap': roadmap})
    
    except Exception as e:
        print(f"Error during roadmap generation: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/job', methods=['POST'])
def get_job():
    try:
        # Ensure a file is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Check if the file is a valid PDF
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Invalid file format, only PDF files are allowed'}), 400
        
        # Process the resume
        resume_text = extract_information(file)
        skills_education = extract_details(resume_text)
        resume_cleaned = skills_education['Skills'] + ' ' + skills_education['Education']
        resume_cleaned = text_cleaning(resume_cleaned)

        # Tokenize and embed resume
        resume_embeddings = []
        tokens = tokenizer(resume_cleaned, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            output = model(**tokens)
        embeddings = output.last_hidden_state.mean(dim=1).numpy()
        resume_embeddings.append(embeddings[0]) 

        # Similarity + Recommendation ##
        similarity_scores = cosine_similarity(job_embeddings_distilbert, resume_embeddings)
        top_jobs = 5
        resume_topjobs = []
        job_similarity_scores = []
        for j in range(len(job_data)):
            score = similarity_scores[j][0]  
            job_similarity_scores.append((j, score))
        job_similarity_scores.sort(key=lambda x: x[1], reverse=True)
        resume_topjobs = job_similarity_scores[:top_jobs]
        
        # Extract website links
        job_data['Website'] = job_data['Company Profile'].apply(extract_website)

        # Construct response JSON
        jobs_output = []
        for job_index, score in resume_topjobs:
            job_title = job_data['Job Title'][job_index]
            company_name = job_data['Company'][job_index]
            website_link = job_data['Website'][job_index]
            jobs_output.append({
                "job_index": job_index + 1,
                # "similarity_score": round(score, 4),
                "job_title": job_title,
                "company_name": company_name,
                "website": website_link
            })

        return jsonify({'topJobs': jobs_output})

    except Exception as e:
        print(f"Error during file upload processing: {e}")
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
