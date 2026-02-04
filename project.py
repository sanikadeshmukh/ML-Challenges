#author - Sanika Prashant Deshmukh, Date - 09-06-2025, Assignment- Implementation of project - Phishing Email Detection.

# import necessary libraries
import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
import random, numpy as np, torch, os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from imblearn.under_sampling import RandomUnderSampler
import os

print("Current Directory:", os.getcwd()) # helps to get current working directory


df = pd.read_csv("mlc_new.csv") # loading the dataset
# === 0. SEED SETUP ===
import random, numpy as np, torch, os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42) # set a fixed seed for reproducibility purpose

# === 1. STRATIFIED 70-15-15 SPLIT BEFORE ANY CLEANING ===
from sklearn.model_selection import train_test_split
# 85% of the data will be used for training and validation, and 15% for testing
def stratified_70_15_15_split(df, label_column='label', random_state=42):
    df_train_val, df_test = train_test_split(df, test_size=0.15, stratify=df[label_column], random_state=random_state)
    df_train, df_val = train_test_split(df_train_val, test_size=0.1765, stratify=df_train_val[label_column], random_state=random_state)
    return {'train': df_train.reset_index(drop=True), 'val': df_val.reset_index(drop=True), 'test': df_test.reset_index(drop=True)}

# Split first
base_split = stratified_70_15_15_split(df)

# === 2. MISSING DATA STRATEGIES ON TRAIN + VAL ONLY ===

def drop_missing_rows(df):
    return df.dropna().reset_index(drop=True) # Drop rows with any missing values


def knn_impute_test(df_train_val, df_test, columns, n_neighbors=5):
    df_test_imputed = df_test.copy() # create a copy of the test set to avoid modifying the original
    for column_name in columns:
        train_texts = df_train_val[column_name].dropna() # columns name to be imputed
        if train_texts.empty or df_test[column_name].isnull().sum() == 0:
            continue
        tfidf = TfidfVectorizer() # preprocess the text data using TF-IDF to convert for text data into numerical format
        tfidf_matrix = tfidf.fit_transform(train_texts)
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine') # using cosine distance is used to find the nearest neighbors
        nn.fit(tfidf_matrix)
        for idx in df_test_imputed[df_test_imputed[column_name].isnull()].index: # for null values in the test set iterate through the indices
            distances, indices = nn.kneighbors(tfidf_matrix[:1], n_neighbors=n_neighbors)
            nearest_idx = indices[0][0]
            imputed_value = train_texts.iloc[nearest_idx] # get the nearest neighbor's value
            df_test_imputed.at[idx, column_name] = imputed_value # assign the imputed value to the test set
    return df_test_imputed


def model_based_impute(df_train_val, df_test, target_columns=['receiver', 'subject']): #rf based imputation
    df_test_imputed = df_test.copy() # create a copy of the test set to avoid modifying the original
    for target_column in target_columns:
        len_col = f'{target_column}_len'
        df_train_val[len_col] = df_train_val[target_column].apply(lambda x: len(str(x)) if pd.notnull(x) else np.nan) # calculate the length of the target column values train_val
        df_test_imputed[len_col] = df_test_imputed[target_column].apply(lambda x: len(str(x)) if pd.notnull(x) else np.nan) # calculate the length of the target column values in test set

        feature_cols = ['urls', len_col] # using urls and length of the target column as features
        train_data = df_train_val[df_train_val[target_column].notnull()].dropna(subset=feature_cols) # filter the train_val data to get rows where target column is not null and drop rows with missing values in feature columns
        predict_data = df_test_imputed[df_test_imputed[target_column].isnull()].dropna(subset=feature_cols)

        if predict_data.empty or train_data.empty:
            df_test_imputed.drop(columns=[len_col], inplace=True) # if there are no rows to predict or train, drop the length column and continue
            continue

        model = RandomForestClassifier(random_state=42) # using Random Forest Classifier for imputation
        model.fit(train_data[feature_cols], train_data[target_column]) # fit the model on the train_val data
        df_test_imputed.loc[predict_data.index, target_column] = model.predict(predict_data[feature_cols]) # predict the target column values for the test set
        df_test_imputed.drop(columns=[len_col], inplace=True)
    return df_test_imputed

# Apply strategies only to train+val
cleaned_data_variants = {}
for strategy in ['drop', 'knn', 'model']: 
    train_val = pd.concat([base_split['train'], base_split['val']]) # Combine train and validation sets
    if strategy == 'drop':
        train_val_clean = drop_missing_rows(train_val) #drop only the rows with missing values in train_val
    elif strategy == 'knn':
        train_val_clean = knn_impute_test(train_val.copy(), base_split['test'].copy(), ['subject', 'receiver']) # KNN for imputation learns from the train_val data and applies it to the test set
    elif strategy == 'model':
        train_val_clean = model_based_impute(train_val.copy(), base_split['test'].copy(), ['subject', 'receiver']) # Model-based imputation learns from the train_val data and applies it to the test set
    # Re-split cleaned train_val
    val_size_ratio = len(base_split['val']) / (len(base_split['train']) + len(base_split['val'])) # test size ratio is calculated based on the original train_val split

    df_train_clean, df_val_clean = train_test_split(
        train_val_clean,
        test_size=val_size_ratio,
        stratify=train_val_clean['label'],
        random_state=42
    )

    cleaned_split = {
        'train': df_train_clean.reset_index(drop=True),
        'val': df_val_clean.reset_index(drop=True),
        'test': base_split['test'].copy()
    }


    cleaned_data_variants[strategy] = cleaned_split
# === 3. FEATURE STRATEGY ===
def apply_text_strategy(df_train, df_test, strategy): #apply text feature represenatation strategies
    if strategy == 'none':
        return np.ones((len(df_train), 1)), np.ones((len(df_test), 1)) # nothing to do, return feature arrays of ones
    elif strategy == 'domain':
        for df in [df_train, df_test]:
            df['body_length'] = df['body'].apply(lambda x: len(str(x))) # get the length of the body text
            df['subject_length'] = df['subject'].apply(lambda x: len(str(x))) # get the length of the subject text
            df['num_exclamations'] = (df['subject'].fillna('') + ' ' + df['body'].fillna('')).apply(lambda x: x.count('!')) # count the number of exclamation marks in subject and body
            df['num_spam_words'] = (df['subject'].fillna('') + ' ' + df['body'].fillna('')).apply(
                lambda x: sum(word in x.lower() for word in ['urgent', 'click', 'verify', 'account', 'password'])) #spam words count
        selected_cols = ['urls', 'body_length', 'subject_length', 'num_exclamations', 'num_spam_words']
        return df_train[selected_cols].values, df_test[selected_cols].values
    elif strategy == 'tfidf':
        train_text = (df_train['sender'].fillna('') + ' ' + df_train['receiver'].fillna('') + ' ' + df_train['subject'].fillna('') + ' ' + df_train['body'].fillna('')) # combine traun sender, receiver, subject and body text into a single text column
        test_text = (df_test['sender'].fillna('') + ' ' + df_test['receiver'].fillna('') + ' ' + df_test['subject'].fillna('') + ' ' + df_test['body'].fillna('')) # combine test sender, receiver, subject and body text into a single text column
        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english', lowercase=True)
        X_train = tfidf.fit_transform(train_text) # fit the TF-IDF vectorizer on the train text
        X_test = tfidf.transform(test_text) # transform the test text using the fitted vectorizer
        return X_train, X_test

    elif strategy == 'bert': 
        model = SentenceTransformer('all-MiniLM-L6-v2')
        def build_input(df):
            return (df['sender'].fillna('') + ' ' + df['receiver'].fillna('') + ' ' +
                    df['subject'].fillna('') + ' ' + df['body'].fillna('')).tolist() # return sender, receiver, subject and body text into a single text column with fillna
        train_text = build_input(df_train) 
        test_text = build_input(df_test)
        X_train = model.encode(train_text, show_progress_bar=True, convert_to_numpy=True) # encode the train text using BERT model
        X_test = model.encode(test_text, show_progress_bar=True, convert_to_numpy=True) # encode the test text using BERT model
        return X_train, X_test

    else:
        raise ValueError("Unknown strategy")
    # === 4. CLASS IMBALANCE STRATEGY ===
def apply_class_imbalance(X_train, y_train, strategy):
    if strategy == 'none':
        return X_train, y_train
    elif strategy == 'class_weight':
        return X_train, y_train  # handled during model hyperparameter tuning
    elif strategy == 'undersample':
        return RandomUnderSampler(random_state=42).fit_resample(X_train, y_train) # undersampling to balance the majority class
    elif strategy == 'smote': #Nitesh V. Chawla, et al., "SMOTE: Synthetic Minority Over-Sampling Technique,”) 
        return SMOTE(random_state=42).fit_resample(X_train, y_train) # SMOTE to generate synthetic samples for the minority class
    else:
        raise ValueError("Unknown imbalance strategy")

# === 5. all strategies pipeline ===

def false_negative_rate(y_true, y_pred):
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp_fn = np.sum(y_true == 1)
    return fn / tp_fn if tp_fn > 0 else 0 # false negative rate calculation

# parameter grids
logreg_grid = { # Logistic Regression hyperparameters
    'C': [1],
    'penalty': ['l2'],
    'solver': ['liblinear'],
    'max_iter': [200],
    'class_weight': [None, 'balanced']
}
rf_grid = { # Random Foest hyperparameters
    'n_estimators': [300],
    'max_depth': [20],
    'min_samples_split': [5],
    'max_features': ['sqrt'],
    'class_weight': [None, 'balanced']
} 
svm_grid = { # SVM hyperparameters
    'C': [1],
    'kernel': ['linear'],
    'class_weight': [None, 'balanced'],
    'max_iter': [1000],
    'probability': [True]
}
#saving the parameter combinations in a list 
logreg_combos = list(ParameterGrid(logreg_grid)) 
rf_combos = list(ParameterGrid(rf_grid))
svm_combos = list(ParameterGrid(svm_grid))

results = []

for missing_strat, split in cleaned_data_variants.items():
    df_train = split['train']
    df_val = split['val']
    df_test = split['test']

    df_test = df_test.copy() # create a copy of the test set to avoid modifying the original
    # fill missing subject values with empty string
    df_test['subject'] = df_test['subject'].fillna('') 
    df_test['receiver'] = df_test['receiver'].fillna('')

    # get labels for train, val, and test sets
    y_train = df_train['label'].values 
    y_val = df_val['label'].values
    y_test = df_test['label'].values

    for feature_strat in ['none', 'domain', 'tfidf']:
      try:
        X_train, X_val = apply_text_strategy(df_train.copy(), df_val.copy(), feature_strat) # apply feature strategy to train and validation sets
        _, X_test = apply_text_strategy(pd.concat([df_train, df_val]), df_test.copy(), feature_strat) # apply feature strategy to test set
        # Convert to dense arrays if TF-IDF
        if hasattr(X_train, "toarray"): 
            X_train = X_train.toarray()
        if hasattr(X_val, "toarray"):
            X_val = X_val.toarray()
        if hasattr(X_test, "toarray"):
            X_test = X_test.toarray()


        #print(f"{missing_strat=}, {feature_strat=}, {imb_strategy=}")
        #print(f"Shapes -> X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
      except Exception as e:
        #print(f"[FEATURE ERROR] feature_strat={feature_strat} failed due to: {e}")
        continue


      for imb_strategy in ['none', 'class_weight', 'undersample', 'smote']:
            X_train_res, y_train_res = apply_class_imbalance(X_train, y_train, imb_strategy) #class imbalance strategy is applied to the training set

            # Convert sparse to dense if needed
            if hasattr(X_train_res, "toarray"):
                X_train_res = X_train_res.toarray()
            if hasattr(X_val, "toarray"):
                X_val = X_val.toarray()
            if hasattr(X_test, "toarray"):
                X_test = X_test.toarray()

            def tune_and_eval(model_name, combos):
                best_score = -1
                best_params = None
                for params in combos:
                    params = params.copy()  # avoid mutating global combos
                    if 'class_weight' in params:
                        params['class_weight'] = 'balanced' if imb_strategy == 'class_weight' else None
                    try:
                        # initialize model based on model_name
                        if model_name == 'LogisticRegression':
                            model = LogisticRegression(**params)
                        elif model_name == 'RandomForest':
                            model = RandomForestClassifier(**params)
                        elif model_name == 'SVM':
                            model = SVC(**params)
                        # fit the model on resampled training data
                        model.fit(X_train_res, y_train_res)
                        # predict on validation set
                        y_val_pred = model.predict(X_val)
                        score = f1_score(y_val, y_val_pred)
                        if score > best_score:
                            best_score = score
                            best_params = params.copy()
                    except Exception as e:
                        #print(f"Tuning failed for {model_name} with params={params} due to: {e}")
                        continue
                if best_params:
                    #print(f"Best params for {model_name} with {missing_strat}, {feature_strat}, {imb_strategy}: {best_params}")
                    pass

                return best_params
            #best parameters for each model are tuned using the tune_and_eval function

            best_lr = tune_and_eval('LogisticRegression', logreg_combos)
            best_rf = tune_and_eval('RandomForest', rf_combos)
            best_svm = tune_and_eval('SVM', svm_combos)

            for model_name, best_params in [
                ('LogisticRegression', best_lr),
                ('RandomForest', best_rf),
                ('SVM', best_svm),
                ('Baseline', None)
            ]:
              try:
                  if model_name == 'Baseline':
                      # Use original non-resampled features for baseline
                      baseline_train_texts = pd.concat([df_train, df_val])
                      baseline_labels = pd.concat([df_train['label'], df_val['label']])
                      baseline_test_texts = df_test.copy()
                      try:
                            # Apply feature strategy to baseline texts for train and test
                          X_baseline_train, X_baseline_test = apply_text_strategy(
                              baseline_train_texts.drop(columns=['label']),
                              baseline_test_texts.drop(columns=['label']),
                              feature_strat
                          )
                          if hasattr(X_baseline_train, "toarray"):
                              X_baseline_train = X_baseline_train.toarray()
                              X_baseline_test = X_baseline_test.toarray()

                          model = DummyClassifier(strategy='most_frequent')
                          # Fit the traun set
                          model.fit(X_baseline_train, baseline_labels)
                          # Predict on the test set
                          y_test_pred = model.predict(X_baseline_test)
                          params_to_log = {'strategy': 'most_frequent'}
                            # Log results
                          results.append({
                              'Model': model_name,
                              'Missing': missing_strat,
                              'Feature': feature_strat,
                              'Imbalance': imb_strategy,
                              'Params': params_to_log,
                              'Accuracy': accuracy_score(y_test, y_test_pred),
                              'Precision': precision_score(y_test, y_test_pred, zero_division=0),
                              'Recall': recall_score(y_test, y_test_pred, zero_division=0),
                              'F1': f1_score(y_test, y_test_pred, zero_division=0),
                              'FNR': false_negative_rate(y_test, y_test_pred)
                          })
                      except Exception as e:
                          #print(f"[BASELINE ERROR] {missing_strat=} {feature_strat=} {imb_strategy=} {e}")
                              pass

                  if best_params is None:
                      # If tuning failed, log the failure
                      results.append({
                          'Model': model_name,
                          'Missing': missing_strat,
                          'Feature': feature_strat,
                          'Imbalance': imb_strategy,
                          'Params': 'Tuning failed',
                          'Accuracy': None,
                          'Precision': None,
                          'Recall': None,
                          'F1': None,
                          'FNR': None
                      })
                      continue

                  #  This now executes properly when best_params exists
                  if model_name == 'LogisticRegression':
                      model = LogisticRegression(**best_params)
                  elif model_name == 'RandomForest':
                      model = RandomForestClassifier(**best_params)
                  elif model_name == 'SVM':
                      model = SVC(**best_params)

                  model.fit(np.vstack((X_train, X_val)), np.concatenate((y_train, y_val))) # concatenate train and validation sets for final training
                  # Predict on the test set
                  y_test_pred = model.predict(X_test)
                  params_to_log = best_params

                  results.append({
                      'Model': model_name,
                      'Missing': missing_strat,
                      'Feature': feature_strat,
                      'Imbalance': imb_strategy,
                      'Params': params_to_log,
                      'Accuracy': accuracy_score(y_test, y_test_pred),
                      'Precision': precision_score(y_test, y_test_pred, zero_division=0),
                      'Recall': recall_score(y_test, y_test_pred, zero_division=0),
                      'F1': f1_score(y_test, y_test_pred, zero_division=0),
                      'FNR': false_negative_rate(y_test, y_test_pred)
                  })
              except Exception as e:
                  #print(f"Evaluation failed for {model_name}: {e}")
                  continue

# save results
results_df = pd.DataFrame(results)
for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'FNR']:
  if metric in results_df.columns:
    results_df[metric] = results_df[metric].round(3)

results_df = results_df.sort_values(by=['Model', 'F1'], ascending=[True, False]).reset_index(drop=True)
results_df.to_csv("all_results_1.csv", index=False)

# === 6. BERT Features ===

bert_results = []

for missing_strat, split in cleaned_data_variants.items():
    df_train = split['train']
    df_val = split['val']
    df_test = split['test'].copy()
    train_val_clean = pd.concat([df_train, df_val])

    # === Apply missing value strategy to test set ===
    # Note: We do not apply missing value strategies to train+val here, as they were already applied in the previous step
    if missing_strat == 'drop':
        df_test = drop_missing_rows(df_test)
    elif missing_strat == 'knn':
        df_test = knn_impute_test(train_val_clean, df_test, ['subject', 'receiver'])
    elif missing_strat == 'model':
        df_test = model_based_impute(train_val_clean, df_test, ['receiver', 'subject'])

    # === Separate features and labels ===
    # x and y for train, val, and test sets
    X_train_df = df_train.drop(columns=['label']).copy()
    y_train = df_train['label'].values

    X_val_df = df_val.drop(columns=['label']).copy()
    y_val = df_val['label'].values

    X_test_df = df_test.drop(columns=['label']).copy()
    y_test = df_test['label'].values

    # === Apply BERT Features ===
    X_train, X_val = apply_text_strategy(X_train_df.copy(), X_val_df.copy(), 'bert') # bert features are applied to the train and validation sets
    _, X_test = apply_text_strategy(pd.concat([X_train_df, X_val_df]), X_test_df.copy(), 'bert') #   bert features are applied to the test set

    for imb_strategy in ['none', 'class_weight', 'undersample', 'smote']:
        X_train_res, y_train_res = apply_class_imbalance(X_train, y_train, imb_strategy)

        def tune_and_eval_bert(model_name, combos):
          best_score = -1
          best_params = None
          for params in combos:
              params = params.copy()
              params['class_weight'] = 'balanced' if imb_strategy == 'class_weight' else None
              try:
                    # Initialize model based on model_name
                  if model_name == 'LogisticRegression':
                      model = LogisticRegression(**params)
                  elif model_name == 'RandomForest':
                      model = RandomForestClassifier(**params)
                  elif model_name == 'SVM':
                      model = SVC(**params)
                      # Fit the model on resampled training data
                  model.fit(X_train_res, y_train_res)
                  # Predict on validation set
                  y_val_pred = model.predict(X_val)
                  score = f1_score(y_val, y_val_pred)
                  if score > best_score:
                      best_score = score
                      best_params = params.copy()
              except Exception as e:
                  #print(f"[TUNE FAIL] {model_name} | {params} | {e}")
                  continue
          return best_params



        best_lr = tune_and_eval_bert('LogisticRegression', logreg_combos)
        #print(f"Best Params for LogisticRegression, {missing_strat}, {imb_strategy}: {best_lr}")

        best_rf = tune_and_eval_bert('RandomForest', rf_combos)
        #print(f"Best Params for RandomForest, {missing_strat}, {imb_strategy}: {best_rf}")

        best_svm = tune_and_eval_bert('SVM', svm_combos)
        #print(f"Best Params for SVM, {missing_strat}, {imb_strategy}: {best_svm}")

        #print("Resampled shapes:", X_train_res.shape, y_train_res.shape)


        for model_name in ['LogisticRegression', 'RandomForest', 'SVM', 'Baseline']:
            try:
                #print("Resampled shapes:", X_train_res.shape, y_train_res.shape)
                # === Baseline Case ===
                if model_name == 'Baseline':
                    model = DummyClassifier(strategy='most_frequent')
                    # Fit the model on the concatenated train and validation sets
                    model.fit(np.vstack((X_train, X_val)), np.concatenate((y_train, y_val)))
                    # Predict on the test set
                    y_test_pred = model.predict(X_test)
                    params_to_log = {'strategy': 'most_frequent'}

                # === Other Models ===
                else:
                    if model_name == 'LogisticRegression':
                        params = best_lr
                        model = LogisticRegression(**params)
                    elif model_name == 'RandomForest':
                        params = best_rf
                        model = RandomForestClassifier(**params)
                    elif model_name == 'SVM':
                        params = best_svm
                        model = SVC(**params)
                    else:
                        continue  # skip unknown

                    if params is None:
                        continue  # tuning failed

                    
                    model.fit(np.vstack((X_train, X_val)), np.concatenate((y_train, y_val))) # fit the model on the concatenated train and validation sets
                    y_test_pred = model.predict(X_test) # predict on the test set
                    params_to_log = params # log the parameters used for the model




                # === Log Final Results ===
                bert_results.append({
                    'Model': model_name,
                    'Missing': missing_strat,
                    'Feature': 'bert',
                    'Imbalance': imb_strategy,
                    'Params': params_to_log,
                    'Accuracy': accuracy_score(y_test, y_test_pred),
                    'Precision': precision_score(y_test, y_test_pred, zero_division=0),
                    'Recall': recall_score(y_test, y_test_pred, zero_division=0),
                    'F1': f1_score(y_test, y_test_pred, zero_division=0),
                    'FNR': false_negative_rate(y_test, y_test_pred)
                })

            except Exception as e:
              #print(f"[EVAL FAIL] {model_name} | {missing_strat} | bert | {imb_strategy} | {e}")
              continue

# Save results
bert_df = pd.DataFrame(bert_results)
for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'FNR']:
    bert_df[metric] = bert_df[metric].round(3)

bert_df = bert_df.sort_values(by=['Model', 'F1'], ascending=[True, False]).reset_index(drop=True)
bert_df.to_csv("bert_results_1.csv", index=False)