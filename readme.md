# Phishing Email Classification 


Name - Sanika Prashant Deshmukh
Class - Machine Learning Challenges
Spring 2025
Project - Phishing Email Detection 

This project implements and compares various strategies for phishing email detection using classical machine learning models. It addresses challenges such as:
- Missing data
- Text feature extraction
- Class imbalance
with Model selection and hyperparameter tuning

It evaluates combinations of missing data handling, feature extraction techniques (TF-IDF, BERT, domain-based), class imbalance strategies (e.g., SMOTE,Undersample,class weight), and classifiers (Logistic Regression, Random Forest, SVM, Dummy Baseline).

---

## 🗃️ Dataset

The project uses a labeled email dataset (`mlc_new.csv`) which includes fields such as:
- `sender`
- `receiver`
- `date`
- `subject`
- `body`
- `urls`
- `label` (0 = legitimate, 1 = phishing)

Ensure the dataset file `mlc_new.csv` is placed in the same directory as the script.

---

## 🛠️ Dependencies

Install the required Python packages before running the project:

```bash
pip install pandas numpy scikit-learn imbalanced-learn sentence-transformers

Main Libraries Used
Library	                Purpose

pandas	                Data loading, manipulation
numpy	                Numerical operations, array handling
scikit-learn            ML models (Logistic Regression, SVM, Random Forest), preprocessing, metrics, parameter tuning
imbalanced-learn	    Handling class imbalance (SMOTE, undersampling, oversampling)
sentence-transformers	Pretrained BERT model for text embeddings (all-MiniLM-L6-v2)
torch	                Backend for sentence-transformers
os	                    File and directory operations
random	                Seeding and randomness control

📂 Files and Structure
project.py – Main script implementing the full pipeline

mlc_new.csv – Input CSV file containing the email dataset

all_results.csv – Output file containing evaluation results for standard features

bert_results.csv – Output file containing evaluation results for BERT features

⚙️ How It Works
Stratified Split: The data is split into train (70%), validation (15%), and test (15%) sets.

Missing Value Handling: Training/validation sets are processed with either:

Drop rows

KNN-based imputation

Model-based imputation (Random Forest)

Feature Extraction:

none: Dummy features

domain: Feature-engineered metrics (lengths, spam indicators)

tfidf: Text vectorization

bert: Sentence embeddings using a pretrained transformer

Class Imbalance Handling:

none, class_weight, undersample, SMOTE

Model Training:

Logistic Regression, Random Forest, SVM, DummyClassifier

Evaluation: Best models are evaluated on the test set and results saved as CSV files


📤 Output
The output is stored in:

all_results.csv – Standard feature results - For baseline the 36 rows repeat and give cells as failed tuning although it has been data at the top the tuning failed are just duplicates, there is nothing missing for baseline

bert_results.csv – BERT feature results

Each row includes:

Model name

Missing value strategy

Feature strategy

Imbalance strategy

Best hyperparameters

Accuracy, Precision, Recall, F1-score, and False Negative Rate on test set



Run the script
Execute your main file:

python project.py


---------------------------------
For extra credit 
use explain_lime.py file - A separate script that uses **LIME** to generate explanations for 3 test predictions.

In the same file structure as our main code
Dependenies - LimeTextExplainer,webbrowser,make_pipeline,TfidfVectorizer,LogisticRegression
Output - Gives .html pages for example with tfidf word counter example.
Run the script - python project_mlc.py
----------------------------------