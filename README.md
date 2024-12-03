# ViaClaro

## Overview

The intricacies of career planning and job finding are quite daunting, especially for the newbies. Individuals often struggle with effectively planning their career trajectory and locating suitable job opportunities. Even though traditional career counselling is in place, it can be quite expensive, time-consuming, and may not provide personalized support. Therefore, we propose a modern Career Coach named ViaClaro (‘via’ meaning path and ‘claro’ meaning clear) with an interactive interface which makes use of modern technologies like Artificial Intelligence, Machine Learning, Natural Language Processing, and Large Language Models to provide personalized career advice based on users’ skills and interests. It will not only suggest potential career paths and available job opportunities based on the users’ skills but will also give a brief on what the career entails and how one can approach it.

## How to Run

You will first need to run `app.py` to start the Flask server. Then, you can access the application by running `index.html`. You will first have to upload your resume and click submit. Two buttons will show up. Clicking the button labelled  `Career Paths` will list the top 5 career paths that you are best suited for. Each career path is a clickable button which gives a short brief on what the profession is and how one might approach it. Clicking the button labelled `Job Recommendations` will list the top 5 job recommendations based on your skills and interests.


## Required Packages

Before running `app.py` and `index.html`, ensure you have installed the necessary packages. Use the commands below to install them.

- **PDF Input and Processing:**
  ```bash
  pip install PyPDF2
  pip install contractions
  pip install pandas

- **Saving/Loading the Model:**
  ```bash
  pip install pickle

- **Getting Career Paths:**
  ```bash
  pip install numpy

- **Generating Road Map:**
  ```bash
  pip install openai==0.28

- **Getting Job Recommendations:**
  ```bash
  pip install transformers
  pip install torch
  pip install scikit-learn

- **Connecting Model to Front-end**
  ```bash
  pip install flask
  pip install flask_cors

- **Combined Command**
  ```bash
  pip install PyPDF2 scikit-learn pandas openai==0.28 contractions transformers torch flask flask_cors numpy pickle