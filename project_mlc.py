#author - Sanika Prashant Deshmukh, Date - 09-06-2025, Assignment- Implementation of project.
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
# Set the working directory to the location of this script
print("Current Directory:", os.getcwd())

# Load the dataset
df = pd.read_csv("mlc_new.csv")
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

set_seed(42)

# === 1. STRATIFIED 70-15-15 SPLIT BEFORE ANY CLEANING ===
from sklearn.model_selection import train_test_split

def stratified_70_15_15_split(df, label_column='label', random_state=42):
    df_train_val, df_test = train_test_split(df, test_size=0.15, stratify=df[label_column], random_state=random_state)
    df_train, df_val = train_test_split(df_train_val, test_size=0.1765, stratify=df_train_val[label_column], random_state=random_state)
    return {'train': df_train.reset_index(drop=True), 'val': df_val.reset_index(drop=True), 'test': df_test.reset_index(drop=True)}

# Split first
base_split = stratified_70_15_15_split(df)

# === 2. MISSING DATA STRATEGIES ON TRAIN + VAL ONLY ===
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier

def drop_missing_rows(df):
    return df.dropna().reset_index(drop=True)


def knn_impute_test(df_train_val, df_test, columns, n_neighbors=5):
    df_test_imputed = df_test.copy()
    for column_name in columns:
        train_texts = df_train_val[column_name].dropna()
        if train_texts.empty or df_test[column_name].isnull().sum() == 0:
            continue
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(train_texts)
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        nn.fit(tfidf_matrix)
        for idx in df_test_imputed[df_test_imputed[column_name].isnull()].index:
            distances, indices = nn.kneighbors(tfidf_matrix[:1], n_neighbors=n_neighbors)
            nearest_idx = indices[0][0]
            imputed_value = train_texts.iloc[nearest_idx]
            df_test_imputed.at[idx, column_name] = imputed_value
    return df_test_imputed


def model_based_impute_test(df_train_val, df_test, target_columns=['receiver', 'subject']):
    df_test_imputed = df_test.copy()
    for target_column in target_columns:
        len_col = f'{target_column}_len'
        df_train_val[len_col] = df_train_val[target_column].apply(lambda x: len(str(x)) if pd.notnull(x) else np.nan)
        df_test_imputed[len_col] = df_test_imputed[target_column].apply(lambda x: len(str(x)) if pd.notnull(x) else np.nan)

        feature_cols = ['urls', len_col]
        train_data = df_train_val[df_train_val[target_column].notnull()].dropna(subset=feature_cols)
        predict_data = df_test_imputed[df_test_imputed[target_column].isnull()].dropna(subset=feature_cols)

        if predict_data.empty or train_data.empty:
            df_test_imputed.drop(columns=[len_col], inplace=True)
            continue

        model = RandomForestClassifier(random_state=42)
        model.fit(train_data[feature_cols], train_data[target_column])
        df_test_imputed.loc[predict_data.index, target_column] = model.predict(predict_data[feature_cols])
        df_test_imputed.drop(columns=[len_col], inplace=True)
    return df_test_imputed

# Apply strategies only to train+val; leave test as-is
cleaned_data_variants = {}
for strategy in ['drop', 'knn', 'model']:
    train_val = pd.concat([base_split['train'], base_split['val']])
    if strategy == 'drop':
        train_val_clean = drop_missing_rows(train_val)
    elif strategy == 'knn':
        train_val_clean = knn_impute_test(train_val.copy(), base_split['test'].copy(), ['subject', 'receiver'])
    elif strategy == 'model':
        train_val_clean = knn_impute_test(train_val.copy(), base_split['test'].copy(), ['subject', 'receiver'])
    # Re-split cleaned train_val
    val_size_ratio = len(base_split['val']) / (len(base_split['train']) + len(base_split['val']))

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
def apply_feature_strategy(df_train, df_test, strategy):
    if strategy == 'none':
        return np.ones((len(df_train), 1)), np.ones((len(df_test), 1))
    elif strategy == 'domain':
        for df in [df_train, df_test]:
            df['body_length'] = df['body'].apply(lambda x: len(str(x)))
            df['subject_length'] = df['subject'].apply(lambda x: len(str(x)))
            df['num_exclamations'] = (df['subject'].fillna('') + ' ' + df['body'].fillna('')).apply(lambda x: x.count('!'))
            df['num_spam_words'] = (df['subject'].fillna('') + ' ' + df['body'].fillna('')).apply(
                lambda x: sum(word in x.lower() for word in ['urgent', 'click', 'verify', 'account', 'password']))
        selected_cols = ['urls', 'body_length', 'subject_length', 'num_exclamations', 'num_spam_words']
        return df_train[selected_cols].values, df_test[selected_cols].values
    elif strategy == 'tfidf':
        train_text = (df_train['sender'].fillna('') + ' ' + df_train['receiver'].fillna('') + ' ' + df_train['subject'].fillna('') + ' ' + df_train['body'].fillna(''))
        test_text = (df_test['sender'].fillna('') + ' ' + df_test['receiver'].fillna('') + ' ' + df_test['subject'].fillna('') + ' ' + df_test['body'].fillna(''))
        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english', lowercase=True)
        X_train = tfidf.fit_transform(train_text)
        X_test = tfidf.transform(test_text)
        return X_train, X_test

    elif strategy == 'bert':
        model = SentenceTransformer('all-MiniLM-L6-v2')
        def build_input(df):
            return (df['sender'].fillna('') + ' ' + df['receiver'].fillna('') + ' ' +
                    df['subject'].fillna('') + ' ' + df['body'].fillna('')).tolist()
        train_text = build_input(df_train)
        test_text = build_input(df_test)
        X_train = model.encode(train_text, show_progress_bar=True, convert_to_numpy=True)
        X_test = model.encode(test_text, show_progress_bar=True, convert_to_numpy=True)
        return X_train, X_test

    else:
        raise ValueError("Unknown strategy")
    # === 4. CLASS IMBALANCE STRATEGY ===
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

def apply_class_imbalance(X_train, y_train, strategy):
    if strategy == 'none':
        return X_train, y_train
    elif strategy == 'class_weight':
        return X_train, y_train  # handled during model init
    elif strategy == 'undersample':
        return RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)
    elif strategy == 'smote':
        return SMOTE(random_state=42).fit_resample(X_train, y_train)
    else:
        raise ValueError("Unknown imbalance strategy")
    
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid
import numpy as np
import pandas as pd

def false_negative_rate(y_true, y_pred):
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp_fn = np.sum(y_true == 1)
    return fn / tp_fn if tp_fn > 0 else 0

# Define parameter grids
logreg_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear'],
    'max_iter': [100, 200, 500],
    'class_weight': [None, 'balanced']
}
rf_grid = {
    'n_estimators': [200,300, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [5,10],
    'max_features': ['sqrt'],
    'class_weight': [None, 'balanced']
}
svm_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear'],
    'class_weight': [None, 'balanced'],
    'max_iter': [1000, 2000, -1],
    'probability': [True]
}

logreg_combos = list(ParameterGrid(logreg_grid))
rf_combos = list(ParameterGrid(rf_grid))
svm_combos = list(ParameterGrid(svm_grid))

results = []
# === GLOBAL TUNING FOR ALL CLASSIFIERS ===
global_best_params = {
    'LogisticRegression': None,
    'RandomForest': None,
    'SVM': None
}
best_scores = {
    'LogisticRegression': -1,
    'RandomForest': -1,
    'SVM': -1
}

# Use default config: drop | domain | none
df_train = cleaned_data_variants['drop']['train']
df_val = cleaned_data_variants['drop']['val']
y_train = df_train['label'].values
y_val = df_val['label'].values

X_train, X_val = apply_feature_strategy(df_train.copy(), df_val.copy(), 'domain')
X_train_res, y_train_res = apply_class_imbalance(X_train, y_train, 'none')

if hasattr(X_train_res, "toarray"):
    X_train_res = X_train_res.toarray()
if hasattr(X_val, "toarray"):
    X_val = X_val.toarray()

# Tune Logistic Regression
for params in logreg_combos:
    try:
        model = LogisticRegression(**params)
        model.fit(X_train_res, y_train_res)
        y_val_pred = model.predict(X_val)
        score = f1_score(y_val, y_val_pred)
        if score > best_scores['LogisticRegression']:
            best_scores['LogisticRegression'] = score
            global_best_params['LogisticRegression'] = params.copy()
    except:
        continue

# Tune Random Forest
for params in rf_combos:
    try:
        model = RandomForestClassifier(**params)
        model.fit(X_train_res, y_train_res)
        y_val_pred = model.predict(X_val)
        score = f1_score(y_val, y_val_pred)
        if score > best_scores['RandomForest']:
            best_scores['RandomForest'] = score
            global_best_params['RandomForest'] = params.copy()
    except:
        continue

# Tune SVM
for params in svm_combos:
    try:
        model = SVC(**params)
        model.fit(X_train_res, y_train_res)
        y_val_pred = model.predict(X_val)
        score = f1_score(y_val, y_val_pred)
        if score > best_scores['SVM']:
            best_scores['SVM'] = score
            global_best_params['SVM'] = params.copy()
    except:
        continue

print("Global Best Params:", global_best_params)

for missing_strat, split in cleaned_data_variants.items():
    df_train = split['train']
    df_val = split['val']
    df_test = split['test']

    df_test = df_test.copy()
    df_test['subject'] = df_test['subject'].fillna('')
    df_test['receiver'] = df_test['receiver'].fillna('')

    y_train = df_train['label'].values
    y_val = df_val['label'].values
    y_test = df_test['label'].values

    for feature_strat in ['none', 'domain', 'tfidf']:
      try:
        X_train, X_val = apply_feature_strategy(df_train.copy(), df_val.copy(), feature_strat)
        _, X_test = apply_feature_strategy(pd.concat([df_train, df_val]), df_test.copy(), feature_strat)
        # 🔧 Convert to dense arrays if TF-IDF (or any other sparse format)
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
            X_train_res, y_train_res = apply_class_imbalance(X_train, y_train, imb_strategy)

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
                        if model_name == 'LogisticRegression':
                            model = LogisticRegression(**params)
                        elif model_name == 'RandomForest':
                            model = RandomForestClassifier(**params)
                        elif model_name == 'SVM':
                            model = SVC(**params)
                        model.fit(X_train_res, y_train_res)
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

            best_lr = global_best_params['LogisticRegression']
            best_rf = global_best_params['RandomForest']
            best_svm = global_best_params['SVM']


            for model_name, best_params in [
                ('LogisticRegression', best_lr),
                ('RandomForest', best_rf),
                ('SVM', best_svm),
                ('Baseline', None)
            ]:
              try:
                  if model_name == 'Baseline':
                      # Use original (non-resampled) features for baseline
                      baseline_train_texts = pd.concat([df_train, df_val])
                      baseline_labels = pd.concat([df_train['label'], df_val['label']])
                      baseline_test_texts = df_test.copy()
                      try:
                          X_baseline_train, X_baseline_test = apply_feature_strategy(
                              baseline_train_texts.drop(columns=['label']),
                              baseline_test_texts.drop(columns=['label']),
                              feature_strat
                          )
                          if hasattr(X_baseline_train, "toarray"):
                              X_baseline_train = X_baseline_train.toarray()
                              X_baseline_test = X_baseline_test.toarray()

                          model = DummyClassifier(strategy='most_frequent')
                          model.fit(X_baseline_train, baseline_labels)
                          y_test_pred = model.predict(X_baseline_test)
                          params_to_log = {'strategy': 'most_frequent'}

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

                  # ✅ This now executes properly when best_params exists
                  if model_name == 'LogisticRegression':
                      model = LogisticRegression(**best_params)
                  elif model_name == 'RandomForest':
                      model = RandomForestClassifier(**best_params)
                  elif model_name == 'SVM':
                      model = SVC(**best_params)

                  model.fit(np.vstack((X_train, X_val)), np.concatenate((y_train, y_val)))
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

# Save results
results_df = pd.DataFrame(results)
for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'FNR']:
  if metric in results_df.columns:
    results_df[metric] = results_df[metric].round(3)

results_df = results_df.sort_values(by=['Model', 'F1'], ascending=[True, False]).reset_index(drop=True)
results_df.to_csv("final_model_results_val_tuned_12.csv", index=False)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

bert_results = []

for missing_strat, split in cleaned_data_variants.items():
    df_train = split['train']
    df_val = split['val']
    df_test = split['test'].copy()
    train_val_clean = pd.concat([df_train, df_val])

    # === Apply missing value strategy to test set ===
    if missing_strat == 'drop':
        df_test = drop_missing_rows(df_test)
    elif missing_strat == 'knn':
        df_test = knn_impute_test(train_val_clean, df_test, ['subject', 'receiver'])
    elif missing_strat == 'model':
        df_test = model_based_impute_test(train_val_clean, df_test, ['receiver', 'subject'])

    # === Separate features and labels ===
    X_train_df = df_train.drop(columns=['label']).copy()
    y_train = df_train['label'].values

    X_val_df = df_val.drop(columns=['label']).copy()
    y_val = df_val['label'].values

    X_test_df = df_test.drop(columns=['label']).copy()
    y_test = df_test['label'].values

    # === Apply BERT Features ===
    X_train, X_val = apply_feature_strategy(X_train_df.copy(), X_val_df.copy(), 'bert')
    _, X_test = apply_feature_strategy(pd.concat([X_train_df, X_val_df]), X_test_df.copy(), 'bert')

    for imb_strategy in ['none', 'class_weight', 'undersample', 'smote']:
        X_train_res, y_train_res = apply_class_imbalance(X_train, y_train, imb_strategy)

        def tune_and_eval_bert(model_name, combos):
          best_score = -1
          best_params = None
          for params in combos:
              params = params.copy()
              params['class_weight'] = 'balanced' if imb_strategy == 'class_weight' else None
              try:
                  if model_name == 'LogisticRegression':
                      model = LogisticRegression(**params)
                  elif model_name == 'RandomForest':
                      model = RandomForestClassifier(**params)
                  elif model_name == 'SVM':
                      model = SVC(**params)
                  model.fit(X_train_res, y_train_res)
                  y_val_pred = model.predict(X_val)
                  score = f1_score(y_val, y_val_pred)
                  if score > best_score:
                      best_score = score
                      best_params = params.copy()
              except Exception as e:
                  #print(f"[TUNE FAIL] {model_name} | {params} | {e}")
                  continue
          return best_params



        best_lr = global_best_params['LogisticRegression']
        best_rf = global_best_params['RandomForest']
        best_svm = global_best_params['SVM']


        for model_name in ['LogisticRegression', 'RandomForest', 'SVM', 'Baseline']:
            try:
                #print("Resampled shapes:", X_train_res.shape, y_train_res.shape)
                # === Baseline Case ===
                if model_name == 'Baseline':
                    model = DummyClassifier(strategy='most_frequent')
                    model.fit(np.vstack((X_train, X_val)), np.concatenate((y_train, y_val)))
                    y_test_pred = model.predict(X_test)
                    params_to_log = {'strategy': 'most_frequent'}

                # === Tuned Model Case ===
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

                    
                    model.fit(np.vstack((X_train, X_val)), np.concatenate((y_train, y_val)))
                    y_test_pred = model.predict(X_test)
                    params_to_log = params




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

# Save and format results
bert_df = pd.DataFrame(bert_results)
for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'FNR']:
    bert_df[metric] = bert_df[metric].round(3)

bert_df = bert_df.sort_values(by=['Model', 'F1'], ascending=[True, False]).reset_index(drop=True)
bert_df.to_csv("final_model_results_bert_val_tuned_12.csv", index=False) 