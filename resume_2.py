import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,accuracy_score
import pickle
from sklearn.utils import shuffle
from display_roadmap import generate_roadmap
import tkinter as tk
from tkinter import filedialog
import PyPDF2

tfidf = TfidfVectorizer(stop_words='english')
le = LabelEncoder()
sys.stdout.reconfigure(encoding='utf-8')

def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)  
    cleanText = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]', ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

def creating_vectors(filename,column):
    # reading files and counting the resumes for each category
    df = pd.read_csv(filename)
    df_shuffled = shuffle(df, random_state=42)
    df = df_shuffled
    #data cleaning of training set 
    df[column] = df[column].apply(lambda x: cleanResume(x))
    # creating vectors for training set
    tfidf.fit(df[column])
    requiredText  = tfidf.transform(df[column])

    return df, requiredText, tfidf

def assigning_categories(df,column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    unique_values = df[column].unique()
    return le, unique_values

def ann(X_train,X_test,y_train,y_test):
    ann_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, activation='relu', solver='adam')
    ann_clf.fit(X_train, y_train)
    y_pred = ann_clf.predict(X_test)
    with open('ann_clf.pkl', 'wb') as ann_file:
        pickle.dump(ann_clf, ann_file)
    return y_pred
    

def naive_bayes(X_train,X_test,y_train,y_test):
    nb_clf = MultinomialNB()
    nb_clf.fit(X_train, y_train)
    y_pred = nb_clf.predict(X_test)
    with open('nb_clf.pkl', 'wb') as ann_file:
        pickle.dump(nb_clf, ann_file)
    return y_pred


def logistic_regression(X_train, X_test, y_train, y_test):
    lr_clf = LogisticRegression(max_iter=1000)  
    lr_clf.fit(X_train, y_train)  
    y_pred = lr_clf.predict(X_test)
    with open('lg_clf.pkl', 'wb') as ann_file:
        pickle.dump(lr_clf, ann_file)
    return y_pred


def compare_models(X_train, X_test, y_train, y_test):
    # Get predictions from each model
    y_pred_ann = ann(X_train, X_test, y_train, y_test)
    y_pred_nb = naive_bayes(X_train, X_test, y_train, y_test)
    y_pred_lr = logistic_regression(X_train, X_test, y_train, y_test)

    # Store metrics for each model
    models = ['ANN', 'Naive Bayes', 'Logistic Regression']
    accuracy = []
    precision = []
    recall = []
    f1_score = []

    for y_pred in [y_pred_ann, y_pred_nb, y_pred_lr]:
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(report['macro avg']['precision'])
        recall.append(report['macro avg']['recall'])
        f1_score.append(report['macro avg']['f1-score'])

    # Data for plotting
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(models))  # the label locations
    width = 0.2  # the width of the bars

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    bars_accuracy = ax.bar(x - width * 1.5, accuracy, width, label='Accuracy', color='steelblue')
    bars_precision = ax.bar(x - width / 2, precision, width, label='Precision', color='cornflowerblue')
    bars_recall = ax.bar(x + width / 2, recall, width, label='Recall', color='lightseagreen')
    bars_f1_score = ax.bar(x + width * 1.5, f1_score, width, label='F1 Score', color='salmon')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison: Accuracy, Precision, Recall, F1 Score')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    # Adding values on top of each bar
    for bars in [bars_accuracy, bars_precision, bars_recall, bars_f1_score]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.savefig('model_comparison.png', bbox_inches='tight')
    plt.close()
   

def select_pdf_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    return file_path

def pdf_to_string(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text
#---------------------------Training Data-----------------------------------#
filename = 'data/Resume.csv'
column_name = 'Resume_str'
df,required_text, tfidf= creating_vectors(filename,column_name)
le,category = assigning_categories(df,'Category')

# X_train,X_test, y_train, y_test = train_test_split(required_text, df['Category'], test_size=0.2, random_state=42)
# compare_models(X_train, X_test, y_train, y_test)



#--------------------------------- Testing --------------------------------#
myresume=""
pdf = select_pdf_file()
if pdf:
    myresume = pdf_to_string(pdf)
with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)
with open('lg_clf.pkl', 'rb') as ann_file:
    lg_clf = pickle.load(ann_file)
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
print(generate_roadmap(top_categories[index-1]))








