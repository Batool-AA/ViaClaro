from resume_2 import select_pdf_file
from resume_2 import pdf_to_string
import pickle
from resume_2 import cleanResume
import numpy as np
from resume_2 import le
from display_roadmap import generate_roadmap


myresume=""

pdf = select_pdf_file()
if pdf:
    myresume = pdf_to_string(pdf)

with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)
with open('lg_clf.pkl', 'rb') as lr_file:
    lg_clf = pickle.load(lr_file)

cleaned_resume = cleanResume(myresume)
input_features = tfidf.transform([cleaned_resume])
predicted_probabilities = lg_clf.predict_proba(input_features)

top_n = 5  
top_n_indices = np.argsort(predicted_probabilities[0])[-top_n:][::-1]  
top_categories = le.inverse_transform(top_n_indices)
top_probabilities = predicted_probabilities[0][top_n_indices]

for category, probability in zip(top_categories, top_probabilities):
    print(f"Category: {category}, Probability: {probability:.4f}")

index = int(input("Choose profession to view the career path: "))
print(top_categories[index-1])
# print(generate_roadmap(top_categories[index-1]))