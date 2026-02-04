# explain_lime.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
import webbrowser

# Load dataset 
df = pd.read_csv("mlc_new.csv")

# Use same split logic as main code
from sklearn.model_selection import train_test_split
df_train_val, df_test = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=42) # 15% test set

# Prepare text
train_texts = (df_train_val['sender'].fillna('') + ' ' +
               df_train_val['receiver'].fillna('') + ' ' +
               df_train_val['subject'].fillna('') + ' ' +
               df_train_val['body'].fillna('')) # Concatenate all text fields and fill NaNs with empty strings for training
test_texts = (df_test['sender'].fillna('') + ' ' +
              df_test['receiver'].fillna('') + ' ' +
              df_test['subject'].fillna('') + ' ' +
              df_test['body'].fillna('')) # Concatenate all text fields and fill NaNs with empty strings for testing
# label y
y_train = df_train_val['label']
y_test = df_test['label']

# TF-IDF + Logistic Regression Approach
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
X_train = vectorizer.fit_transform(train_texts) # Fit vectorizer on training texts
model = LogisticRegression(C=1, penalty='l2', solver='liblinear', class_weight='balanced', max_iter=200)
model.fit(X_train, y_train) # Train the model

# Make LIME pipeline
pipeline = make_pipeline(vectorizer, model)
explainer = LimeTextExplainer(class_names=["Legitimate", "Phishing"])

# Explain 3 emails
for idx in [5, 10, 15]:
    email_text = test_texts.iloc[idx]
    true_label = y_test.iloc[idx]
    pred = pipeline.predict([email_text])[0] # Get prediction for the email
    print(f"\n Email {idx} — True: {true_label}, Predicted: {pred}")

    explanation = explainer.explain_instance(email_text, pipeline.predict_proba, num_features=10)
    html_file = f"lime_email_{idx}.html"
    explanation.save_to_file(html_file)
    webbrowser.open(html_file) # Open the explanation in the default web browser
# Note: The above code assumes that the dataset 'mlc_new.csv' is in the same directory as this script.
