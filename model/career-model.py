from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import matplotlib.pyplot as plt
import numpy as np
from functions import creating_vectors
from functions import assigning_categories
from functions import combine_files
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns


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
    with open('nb_clf.pkl', 'wb') as nb_file:
        pickle.dump(nb_clf, nb_file)
    return y_pred


def logistic_regression(X_train, X_test, y_train, y_test):
    lr_clf = LogisticRegression(max_iter=1000)  
    lr_clf.fit(X_train, y_train)  
    y_pred = lr_clf.predict(X_test)
    with open('lg_clf.pkl', 'wb') as lr_file:
        pickle.dump(lr_clf, lr_file)
    return y_pred


def compare_models(X_train, X_test, y_train, y_test):
    y_pred_ann = ann(X_train, X_test, y_train, y_test)
    y_pred_nb = naive_bayes(X_train, X_test, y_train, y_test)
    y_pred_lr = logistic_regression(X_train, X_test, y_train, y_test)

    models = ['ANN', 'Naive Bayes', 'Logistic Regression']
    accuracy = []
    precision = []
    recall = []
    f1_score = []

    for y_pred in [y_pred_ann, y_pred_nb, y_pred_lr]:
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(report['macro avg']['precision'])
        recall.append(report['macro avg']['recall'])
        f1_score.append(report['macro avg']['f1-score'])

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(models))  
    width = 0.2  

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_accuracy = ax.bar(x - width * 1.5, accuracy, width, label='Accuracy', color='steelblue')
    bars_precision = ax.bar(x - width / 2, precision, width, label='Precision', color='cornflowerblue')
    bars_recall = ax.bar(x + width / 2, recall, width, label='Recall', color='lightseagreen')
    bars_f1_score = ax.bar(x + width * 1.5, f1_score, width, label='F1 Score', color='salmon')

    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison: Accuracy, Precision, Recall, F1 Score')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    for bars in [bars_accuracy, bars_precision, bars_recall, bars_f1_score]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.savefig('model_comparison.png', bbox_inches='tight')
    plt.close()

#---------------------------------- Training -------------------------------------#
filename1 = 'data/Resume.csv'
filename2 = 'data/UpdatedResumeDataSet.csv'
column_name1 = 'Resume_str' 
column_name2 = 'Category'  
df_combined = combine_files(filename1, filename2, column_name1, column_name2)
required_text, tfidf = creating_vectors(df_combined, column_name1)
le, category = assigning_categories(df_combined, column_name2)


X_train,X_test, y_train, y_test = train_test_split(required_text, df_combined['Category'], test_size=0.2, random_state=42, stratify=df_combined['Category'])

#---------------------------------- Training + Comparing Models --------------------------------------#
compare_models(X_train, X_test, y_train, y_test)

#------------------------ Testing and Training Plotting -----------------------------#
train_df = pd.DataFrame({'Category': le.inverse_transform(y_train)})
test_df = pd.DataFrame({'Category': le.inverse_transform(y_test)})

unique_categories = train_df['Category'].unique()
colors = sns.color_palette("husl", len(unique_categories))

plt.figure(figsize=(15, 5))
sns.countplot(data=train_df, x='Category', hue='Category', palette=colors, legend=False, order=unique_categories)
plt.title('Training Categories Distribution')
plt.xticks(rotation=90)
plt.savefig('training_category_count_plot.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(15, 5))
sns.countplot(data=test_df, x='Category', hue='Category', palette=colors, legend=False, order=unique_categories)
plt.title('Testing Categories Distribution')
plt.xticks(rotation=90)
plt.savefig('testing_category_count_plot.png', bbox_inches='tight')
plt.close()