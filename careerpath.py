from functions import select_pdf_file
from functions import pdf_to_string, extract_information
import pickle
from functions import cleanResume
import numpy as np
from functions import generate_roadmap

myresume=""

pdf = select_pdf_file()
if pdf:
    myresume = extract_information(pdf)

with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)
with open('ann_clf.pkl', 'rb') as ann_file:
    ann_clf = pickle.load(ann_file)
with open('label_encoder.pkl', 'rb') as le_file:
    le_loaded = pickle.load(le_file)

cleaned_resume = cleanResume(myresume)
input_features = tfidf.transform([cleaned_resume])
predicted_probabilities = ann_clf.predict_proba(input_features)

top_n = 5  
top_n_indices = np.argsort(predicted_probabilities[0])[-top_n:][::-1]  
top_categories = le_loaded.inverse_transform(top_n_indices)
top_probabilities = predicted_probabilities[0][top_n_indices]

for category, probability in zip(top_categories, top_probabilities):
    print(f"Category: {category}, Probability: {probability:.4f}")

index = int(input("Choose profession to view the career path: "))
print(top_categories[index-1])
print(generate_roadmap(top_categories[index-1]))