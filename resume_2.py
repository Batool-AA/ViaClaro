import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
# import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import pickle

tfidf = TfidfVectorizer(stop_words='english')
le = LabelEncoder()

sys.stdout.reconfigure(encoding='utf-8')

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# reading files and counting the resumes for each category
df = pd.read_csv('data/Resume.csv')

#data cleaning
df['Resume_str'] = df['Resume_str'].apply(lambda x: cleanResume(x))


#assigning index to every category
le.fit(df['Category'])
df['Category'] = le.transform(df['Category'])
df.Category.unique()

# creating vectors 
tfidf.fit(df['Resume_str'])
requredTaxt  = tfidf.transform(df['Resume_str'])

with open('tfidf.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidf, tfidf_file)

# #training and test data
X_train, X_test, y_train, y_test = train_test_split(requredTaxt, df['Category'], test_size=0.2, random_state=42)

# #using multimonial naive bayes
nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)
y_pred = nb_clf.predict(X_test)
print("Naive Bayes:")
print(classification_report(y_test,y_pred))


#using ann
# Instantiate the MLPClassifier (Multi-Layer Perceptron)
ann_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, activation='relu', solver='adam')

# Fit the model on the training data
ann_clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = ann_clf.predict(X_test)

# Evaluate accuracy
print("ANN:")
print(classification_report(y_test,y_pred))
with open('ann_clf.pkl', 'wb') as ann_file:
    pickle.dump(ann_clf, ann_file)

#testing 
# myresume = """ freelance graphic designer highlights web print design skills software visual elements image photo layout typography color management image file prep retouching resizing formatting packaging press check software adobe creative suite photoshop in design illustrator acrobat creative cloud tumblr square space word press basic html css microsoft office word excel power point outlook mac os 10 11 experience freelance graphic designer 05 2016 current city state influential graphic designer for a high end jewelry company in new york city where i brought originality curiosity enthusiasm a ountability and grit to the table everyday for nearly four years started my own jewelry company called wyndesigns out of college the brand encouraged women to wear their name proudly gia a redited gemologist professional played an instrumental role in the development of the rollins college women s lacrosse program captain senior year website www lgoodwyn com rollins college portfolio design experience created an icon logo for evolve space a company that provides open space environments where professionals and organizations can work build and pursue their visions and missions in a modern collaborative space working directly with the founder i su essfully brought his vision to life providing him with a multi functional icon fit for different web and print scenarios graphic designer assistant 04 2012 01 2016 company name city state lead graphic designer for the company s madison avenue jewelry boutique owned by new york city philanthropist ann ziff produced all advertisements exhibition invitations and marketing materials executing multiple simultaneously under demanding deadlines ran and oversaw the production process for all of the print web projects listed above executing multiple jobs simultaneously under demanding deadlines worked individually as well as collaboratively with the boutique manager offsite art directors producers photographers and printers presented all assets to the boutique owner in a clear and professional manner organized photo shoots prepared pieces and their respective set ups prior to shoot directed the photographers on product placement layout during each shoot updated and maintained the boutique s website and social media outlets instagram facebook twitter yext as event coordinator i managed logistics with offsite organizations cohosting each event coordinated caterer decorations and handled rsvp lists ran all jewelry production fabrication and oversaw the shipping of raw materials and repairs for tamsen z frequently communicated and assisted with boutique cliental directly conducted and directed store inventory updated jewelry database gemini handled all gift purchases and distribution for family friends clients and members of several philanthropic boards which included the metropolitan opera lincoln center and the metropolitan museum of art assisted with personal correspondence edited met opera acknowledgement letters written on ann s behalf and communicated these revisions to their development office mail and phone management scheduling travel arrangements all of which required excellent verbal and communication skills owner designer wyndesigns october 2011 designed and sold bespoke key chains for the line which i created and managed sold work at amethyst a jewelry boutique in bethesda maryland donated pieces to charity auctions such as the children s hospital holiday gala in washington d c experienced with sketching hand sawing welding soldering annealing forging bezel setting sanding and polishing 11 2011 03 2012 city state handled custom client orders worked with customers assisted with trunk shows updated the website created beaded jewelry for boutique education 2011 bachelors degree rollins college studio art city state gpa gpa 3 13 national society of collegiate scholars and phi eta sigma freshman year captain of the women s lacrosse team education chairman of kappa kappa gamma studio art gpa 3 13 national society of collegiate scholars and phi eta sigma freshman year captain of the women s lacrosse team education chairman of kappa kappa gamma 2013 gemological institute of america completed courses in jewelry essentials and colored stone essentials diamond essentials 92nd y new york city may 2010 university of edinburgh college of art city scotland completed courses in metal sculpture and wire jewelry summer program 2009 rhode island school of design city state completed introduction to metal jewelry course summer program skills adobe creative suite acrobat photo photoshop advertisements art avenue c color com communication skills css client clients database functional graphic designer basic html illustrator image inventory layout letters logistics logo mac os marketing materials materials excel mail microsoft office office outlook power point word packaging press print design printers repairs scheduling sculpture shipping sketching soldering phone travel arrangements typography vision website welding written
# """
# with open('tfidf.pkl', 'rb') as tfidf_file:
#     tfidf = pickle.load(tfidf_file)
# with open('ann_clf.pkl', 'rb') as ann_file:
#     ann_clf = pickle.load(ann_file)

# # Clean the input resume
# cleaned_resume = cleanResume(myresume)

# # Transform the cleaned resume using the trained TfidfVectorizer
# input_features = tfidf.transform([cleaned_resume])

# # Make the prediction using the loaded classifier
# prediction_id = ann_clf.predict(input_features)[0]

# print(prediction_id)
# category_name = le.inverse_transform([prediction_id])[0]
# print("Category Name: ", category_name)









