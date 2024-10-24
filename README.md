# ViaClaro

## Overview

The intricacies of career planning and job finding are quite daunting, especially for the newbies. Individuals often struggle with effectively planning their career trajectory and locating suitable job opportunities. Even though traditional career counselling is in place, it can be quite expensive, time-consuming, and may not provide personalized support. Therefore, we propose a Career Coach Chatbot named Via Claro (‘via’ meaning path and ‘claro’ meaning clear) that can assist users in defining a potential career path and in finding the right job opportunity.

This tool makes use of modern technologies like Artificial Intelligence, Machine Learning, Natural Language Processing, and Large Language Models to provide personalized career advice based on users’ skills and interests. The chatbot will not only suggest potential career paths and available job opportunities based on the users’ skills but will also suggest an optimized road map to achieve the users’ career goals.

## How to Run

You need to run the `main.py` file. You will be asked to upload a pdf file via a window pop-up. The chatbot will then process the uploaded file and provide you with a personalized career advice. You can then choose any number between 1 and 5 to display the road map. The chatbot will then display the road map based on your choice if the open-ai api key has been given.  


## Required Packages

Before running `main.py`, ensure you have installed the necessary packages. Use the commands below to install them.

- **PDF Input and Processing:**
  ```bash
  pip install PyPDF2

- **Saving/Loading the Model:**
  ```bash
  pip install pickle

- **Getting the Top 5 Categories:**
  ```bash
  pip install numpy

- **Generating Road Map:**
  ```bash
  pip install openai == 0.28


## Additonal Packages

Before running `model.py`, ensure you have installed the necessary packages. Use the commands below to install them.

- **Running Built-in Models:**
  ```bash
  pip install scikit-learn

- **Plotting Graphs:**
  ```bash
    pip install pandas
    pip install seaborn

