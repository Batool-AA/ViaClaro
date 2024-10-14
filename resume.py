import os
import pdfplumber
import re
import csv

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
        
        # Clean up skills (remove empty and extra spaces)
        skills_list = [skill.strip() for skill in skills_list if skill.strip()]
        
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

def process_all_folders(path, output_csv):
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
                process_pdfs(folder_path, csv_writer)

# Specify the folder containing the subfolders and PDFs
folder = "data/data"

# Specify the output CSV file
output_csv = "output_skills.csv"

# Process all folders and store the results in the CSV
process_all_folders(folder, output_csv)
