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
    # Instantiate the MLPClassifier (Multi-Layer Perceptron)
    ann_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, activation='relu', solver='adam')

    # Fit the model on the training data
    ann_clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = ann_clf.predict(X_test)
    with open('ann_clf.pkl', 'wb') as ann_file:
        pickle.dump(ann_clf, ann_file)

    # Evaluating the model performance
    # print("ANN:")
    # print(classification_report(y_test,y_pred))
    return y_pred
    

def naive_bayes(X_train,X_test,y_train,y_test):
    nb_clf = MultinomialNB()
    nb_clf.fit(X_train, y_train)
    y_pred = nb_clf.predict(X_test)
    with open('nb_clf.pkl', 'wb') as ann_file:
        pickle.dump(nb_clf, ann_file)
    # print("Naive Bayes:")
    # print(classification_report(y_test,y_pred))
    return y_pred


def logistic_regression(X_train, X_test, y_train, y_test):
    # Initialize the Logistic Regression classifier
    lr_clf = LogisticRegression(max_iter=1000)  
    lr_clf.fit(X_train, y_train)  
    # Make predictions on the test data
    y_pred = lr_clf.predict(X_test)
    with open('lg_clf.pkl', 'wb') as ann_file:
        pickle.dump(lr_clf, ann_file)
    # Print the classification report
    # print("Logistic Regression:")
    # print(classification_report(y_test, y_pred))
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


# print(df['Category'].unique())
le,category = assigning_categories(df,'Category')
#binary file to store the state of training data
# with open('tfidf.pkl', 'wb') as tfidf_file:
#     pickle.dump(tfidf, tfidf_file)
# X_train = required_text  
# y_train = df['Category']  



#--------------------------Testing Data-------------------------------------#
# filename_test = 'data/UpdatedResumeDataSet.csv'
# df_test = pd.read_csv(filename_test)
# df_test['Resume'] = df_test['Resume'].apply(lambda x: cleanResume(x))

# # Load the same TF-IDF vectorizer used in training
# with open('tfidf.pkl', 'rb') as tfidf_file:
#     tfidf = pickle.load(tfidf_file)

# # Transform the test data using the same TF-IDF vectorizer
# X_test = tfidf.transform(df_test['Resume']) 
# y_test = df_test['Category'] 


#training and test data
# X_train,X_test, y_train, y_test = train_test_split(required_text, df['Category'], test_size=0.2, random_state=42)
# # naive_bayes(X_train,X_test,y_train,y_test)
# # ann(X_train,X_test,y_train,y_test)
# # logistic_regression(X_train, X_test, y_train, y_test)
# compare_models(X_train, X_test, y_train, y_test)

#for plotting data visually
# train_df = pd.DataFrame({'Category': le.inverse_transform(y_train)})
# test_df = pd.DataFrame({'Category': le.inverse_transform(y_test)})

# unique_categories = train_df['Category'].unique()
# colors = sns.color_palette("husl", len(unique_categories))

# plt.figure(figsize=(15, 5))
# sns.countplot(data=train_df, x='Category',palette=colors)
# plt.title('Training Categories Distribution')
# plt.xticks(rotation=90)
# plt.savefig('training_category_count_plot.png', bbox_inches='tight')
# plt.close()

# plt.figure(figsize=(15, 5))
# sns.countplot(data=test_df, x='Category', palette=colors)
# plt.title('Testing Categories Distribution')
# plt.xticks(rotation=90)
# plt.savefig('testing_category_count_plot.png', bbox_inches='tight')
# plt.close()




#testing 
myresume=""
pdf = select_pdf_file()
if pdf:
    myresume = pdf_to_string(pdf)

# myresume = """ freelance graphic designer highlights web print design skills software visual elements image photo layout typography color management image file prep retouching resizing formatting packaging press check software adobe creative suite photoshop in design illustrator acrobat creative cloud tumblr square space word press basic html css microsoft office word excel power point outlook mac os 10 11 experience freelance graphic designer 05 2016 current city state influential graphic designer for a high end jewelry company in new york city where i brought originality curiosity enthusiasm a ountability and grit to the table everyday for nearly four years started my own jewelry company called wyndesigns out of college the brand encouraged women to wear their name proudly gia a redited gemologist professional played an instrumental role in the development of the rollins college women s lacrosse program captain senior year website www lgoodwyn com rollins college portfolio design experience created an icon logo for evolve space a company that provides open space environments where professionals and organizations can work build and pursue their visions and missions in a modern collaborative space working directly with the founder i su essfully brought his vision to life providing him with a multi functional icon fit for different web and print scenarios graphic designer assistant 04 2012 01 2016 company name city state lead graphic designer for the company s madison avenue jewelry boutique owned by new york city philanthropist ann ziff produced all advertisements exhibition invitations and marketing materials executing multiple simultaneously under demanding deadlines ran and oversaw the production process for all of the print web projects listed above executing multiple jobs simultaneously under demanding deadlines worked individually as well as collaboratively with the boutique manager offsite art directors producers photographers and printers presented all assets to the boutique owner in a clear and professional manner organized photo shoots prepared pieces and their respective set ups prior to shoot directed the photographers on product placement layout during each shoot updated and maintained the boutique s website and social media outlets instagram facebook twitter yext as event coordinator i managed logistics with offsite organizations cohosting each event coordinated caterer decorations and handled rsvp lists ran all jewelry production fabrication and oversaw the shipping of raw materials and repairs for tamsen z frequently communicated and assisted with boutique cliental directly conducted and directed store inventory updated jewelry database gemini handled all gift purchases and distribution for family friends clients and members of several philanthropic boards which included the metropolitan opera lincoln center and the metropolitan museum of art assisted with personal correspondence edited met opera acknowledgement letters written on ann s behalf and communicated these revisions to their development office mail and phone management scheduling travel arrangements all of which required excellent verbal and communication skills owner designer wyndesigns october 2011 designed and sold bespoke key chains for the line which i created and managed sold work at amethyst a jewelry boutique in bethesda maryland donated pieces to charity auctions such as the children s hospital holiday gala in washington d c experienced with sketching hand sawing welding soldering annealing forging bezel setting sanding and polishing 11 2011 03 2012 city state handled custom client orders worked with customers assisted with trunk shows updated the website created beaded jewelry for boutique education 2011 bachelors degree rollins college studio art city state gpa gpa 3 13 national society of collegiate scholars and phi eta sigma freshman year captain of the women s lacrosse team education chairman of kappa kappa gamma studio art gpa 3 13 national society of collegiate scholars and phi eta sigma freshman year captain of the women s lacrosse team education chairman of kappa kappa gamma 2013 gemological institute of america completed courses in jewelry essentials and colored stone essentials diamond essentials 92nd y new york city may 2010 university of edinburgh college of art city scotland completed courses in metal sculpture and wire jewelry summer program 2009 rhode island school of design city state completed introduction to metal jewelry course summer program skills adobe creative suite acrobat photo photoshop advertisements art avenue c color com communication skills css client clients database functional graphic designer basic html illustrator image inventory layout letters logistics logo mac os marketing materials materials excel mail microsoft office office outlook power point word packaging press print design printers repairs scheduling sculpture shipping sketching soldering phone travel arrangements typography vision website welding written
# """
with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)
with open('lg_clf.pkl', 'rb') as ann_file:
    lg_clf = pickle.load(ann_file)

# Clean the input resume
cleaned_resume = cleanResume(myresume)

# Transform the cleaned resume using the trained TfidfVectorizer
input_features = tfidf.transform([cleaned_resume])

# Get the predicted probabilities for each category
predicted_probabilities = lg_clf.predict_proba(input_features)

# Get the indices of the top N categories
top_n = 5  
top_n_indices = np.argsort(predicted_probabilities[0])[-top_n:][::-1]  # Sort and get indices of top N

# Get the category names corresponding to the top N indices
top_categories = le.inverse_transform(top_n_indices)

# Get the probabilities of the top N categories
top_probabilities = predicted_probabilities[0][top_n_indices]

for category, probability in zip(top_categories, top_probabilities):
    print(f"Category: {category}, Probability: {probability:.4f}")


index = int(input("Choose profession to view the career path: "))
print(top_categories[index-1])
print(generate_roadmap(top_categories[index-1]))








