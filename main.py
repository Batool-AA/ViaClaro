from sklearn.model_selection import train_test_split
from functions import creating_vectors
from functions import assigning_categories
from model import compare_models
from functions import select_pdf_file
from functions import pdf_to_string
import pickle
from functions import cleanResume
import numpy as np
from functions import le
from functions import generate_roadmap

#---------------------------------- Training -------------------------------------#
filename = 'data/Resume.csv'
column_name = 'Resume_str'
df,required_text, tfidf= creating_vectors(filename,column_name)
le,category = assigning_categories(df,'Category')

# X_train,X_test, y_train, y_test = train_test_split(required_text, df['Category'], test_size=0.2, random_state=42)

#---------------------------------- Compare --------------------------------------#
# compare_models(X_train, X_test, y_train, y_test)

#---------------------------------- Prediction -----------------------------------#
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