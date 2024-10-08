import os
import pdfplumber
import re



def extract_information(full_text):
    normalize = full_text.lower()
    skills_pattern = re.compile(r'skills\s*[\n\r]+(.*)', re.IGNORECASE)
    skills_list=[]
    education_list=[]
    # Extract Skills Section
    skills_match = skills_pattern.search(normalize)
    if skills_match:
        skills_section = skills_match.group(1)  # Everything after 'Skills:'
        
        # Split skills by common delimiters (comma, semicolon, newlines, bullets)
        skills_list = re.split(r',|;|\n|•', skills_section)
        
        # Clean up skills (remove empty and extra spaces)
        skills_list = [skill.strip() for skill in skills_list if skill.strip()]
    # Extract Education Section
    education_pattern = re.compile(r'education\s*.*\s*(.*?)(?=\n\s*\n|\Z)', re.IGNORECASE)
    education_match = education_pattern.search(normalize)
    if education_match:
        education_section = education_match.group(1)  # Everything after 'Education:'
        
        # Split education entries by new lines
        education_list = re.split(r'\n', education_section)
        
        # Clean up education entries (remove empty and extra spaces)
        education_list = [edu.strip() for edu in education_list if edu.strip()]
    return {
        'skills':skills_list,
        'education':education_list
    }

def process_pdfs(folder_path):
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
            folder_data[file_name] = {'skills':data['skills'],'education':data['education']}
            print(folder_data)
        
    # return folder_data



def process_all_folders(path):
    all_folders_data = {}  # Dictionary to store data for all folders
    
    # Iterate over all subfolders inside the data directory
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        
        if os.path.isdir(folder_path):  # Ensure it's a directory
            # Process the PDF files in this folder
            folder_data = process_pdfs(folder_path)
            
    #         # Store the folder data in the main dictionary
    #         all_folders_data[folder_name] = folder_data
    
    # return all_folders_data


folder = "data/data"
process_all_folders(folder)