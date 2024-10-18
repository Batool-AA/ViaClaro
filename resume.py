import os
import pdfplumber
import re
import pandas as pd
import csv
import string


def clean_skills(skills):
    # Convert to lowercase
    skills = skills.lower()
    
    # Remove punctuation
    skills = skills.translate(str.maketrans('', '', string.punctuation))
    
    # Remove digits
    skills = re.sub(r'\d+', '', skills)
    
    # Split the string into words
    words = skills.split()
    
    return words

def process_excel(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Initialize an empty dictionary
    skills_dict = {}
    job_skills=[]
    
    # Iterate over the rows
    for index, row in df.iterrows():
        link = row['job_link']  # Assuming the first column is 'Link'
        skills = row['job_skills']  # Assuming the second column is 'Skills'
        
        # Clean the skills
        cleaned_skills = clean_skills(skills)
        
        # Add to dictionary
        skills_dict[link] = cleaned_skills
        job_skills.append(cleaned_skills)
    
    return skills_dict,job_skills



def extract_information(full_text):
    normalize = full_text.lower()
    skills_pattern = re.compile(r'skills\s*[\n\r]+(.*)', re.IGNORECASE)
    skills_list = []
    education_list = []
    
    # Extract Skills Section
    skills_match = skills_pattern.search(normalize)
    if skills_match:
        skills_section = skills_match.group(1)  # Everything after 'Skills:'
        
        # Split skills by common delimiters (comma, semicolon, newlines, bullets)
        skills_list = re.split(r',|;|\n|•', skills_section)
        
        # Function to remove special characters and clean each skill
        def clean_skill(skill):
            # Remove punctuation, special characters, and digits
            skill = skill.translate(str.maketrans('', '', string.punctuation + string.digits))
            
            # Remove non-alphabetic characters (optional)
            skill = re.sub(r'[^a-zA-Z\s]', '', skill)
            
            # Strip extra spaces and return cleaned skill
            return skill.strip()

        # Clean up skills: remove empty, extra spaces, special characters, and digits
        skills_list = [clean_skill(skill) for skill in skills_list if clean_skill(skill)]
        

        
    return {
        'skills': skills_list,
        'education': education_list
    }

def process_pdfs(folder_path, csv_writer):
    folder_data = {}  # Dictionary to store data for this folder
    
    # Iterate over all PDF files in the current folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):  # Ensure we only process PDF files
            pdf_path = os.path.join(folder_path, file_name)
            
            # Open and extract text from the PDF
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    full_text += page.extract_text()

            # Extract information from the full text of the PDF
            data = extract_information(full_text)
            
            # Store the file name and skills in the CSV
            if data['skills']:
                csv_writer.writerow([file_name, ', '.join(data['skills'])])
                folder_data[file_name] = data['skills']
    return folder_data
            

def process_all_folders(path, output_csv):
    resume_skills=[]
    # Open the CSV file to write the results
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        
        # Write the header
        csv_writer.writerow(['File Name', 'Skills'])
        
        # Iterate over all subfolders inside the data directory
        for folder_name in os.listdir(path):
            folder_path = os.path.join(path, folder_name)
            
            if os.path.isdir(folder_path):  # Ensure it's a directory
                # Process the PDF files in this folder and write to CSV
                dictionary=process_pdfs(folder_path, csv_writer)
                for i in dictionary:
                    resume_skills.append(dictionary[i])
    return resume_skills

def vocabulary(resume,job):
    vocab=[]
    for i in resume:
        for j in i:
            if j not in vocab:
                vocab.append(j)
    
    for i in job:
        for j in i:
            if j not in vocab:
                vocab.append(j)
    return vocab



# Specify the folder containing the subfolders and PDFs
folder = "data/data"

# Specify the output CSV file
output_csv = "output_skills.csv"

file_jobs = "data/Jobs/jobs1.xlsx"
job_dict,job_skills = process_excel(file_jobs)

# Process all folders and store the results in the CSV
# resume_skills = process_all_folders(folder, output_csv)
vocab = vocabulary("",job_skills)
print("Vocabulary: ", vocab)

