
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

#loading dataset
df = pd.read_csv("/Users/anju/Desktop/diabetes_dataset_with_notes.csv")  
df.head()

# Data Cleaning & Preprocessing
# Handle missing values (drop or impute)
missing_values=df.isnull().sum()
print("Missing values in each column: ")
print(missing_values)


categorical_cols=['gender', 'location', 'smoking_history']
for col in categorical_cols:
    df[col]=df[col].astype('category')

print(df.dtypes)
df.describe(include='all')

#data analysis
sns.countplot(x='diabetes', data=df)
plt.title('Distribution of Diabetes Status')
plt.show()

# Select only numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=[np.number])

# Correlation Heatmap 
if numeric_df.shape[1] >= 4:
    plt.figure(figsize=(10,8))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap of Numeric Features')
    plt.tight_layout()
    plt.show()
else:
    print('Not enough numeric columns for a correlation heatmap.')

# Pair Plot for a subset of the numeric variables to visualize pairwise relationships
sns.pairplot(numeric_df.dropna())
plt.suptitle('Pair Plot of Numeric Features', y=1.02)
plt.show()

# Histogram for numeric distributions - age, bmi, hbA1c_level
plt.figure(figsize=(12,4))
for i, col in enumerate(['age', 'bmi', 'hbA1c_level']):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Count plot for categorical variables such as gender
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='gender', palette='pastel')
plt.title('Count Plot of Gender')
plt.show()

# Grouped Bar Plot for race distribution across diabetes statuses
race_cols = ['race:AfricanAmerican', 'race:Asian', 'race:Caucasian', 'race:Hispanic', 'race:Other']
race_df = df[race_cols + ['diabetes']].copy()
race_df = race_df.melt(id_vars='diabetes', var_name='Race', value_name='Count')

plt.figure(figsize=(10,6))
sns.barplot(data=race_df, x='Race', y='Count', hue='diabetes', palette='muted')
plt.title('Race Distribution Grouped by Diabetes Status')
plt.show()

# Box Plot for BMI grouped by diabetes status
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='diabetes', y='bmi', palette='Set2')
plt.title('BMI Distribution by Diabetes Status')
plt.show()

sns.histplot(df['blood_glucose_level'], kde=True)
plt.title("Distribution of Glucose")
plt.show()

import scipy.stats as stats

stats.probplot(df['blood_glucose_level'], dist="norm", plot=plt)
plt.title("Q-Q Plot for Glucose")
plt.show()




results = {}

X_original = df.drop(columns=['diabetes', 'clinical_notes'])  # Drop clinical_notes if it's raw text
y = df['diabetes']

def preprocess_for_naive_bayes(X):
    """ Keep only numeric columns for Naive Bayes """
    return X.select_dtypes(include=[np.number])

def preprocess_for_logistic(X):
    """ One-hot encode categorical columns for Logistic Regression """
    return pd.get_dummies(X, drop_first=True)

def preprocess_for_tree(X):
    """ Tree-based models can handle one-hot and ordinal without scaling """
    return pd.get_dummies(X, drop_first=True)

def preprocess_and_scale(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Model: Naive Bayes Not included in the report
X_nb = preprocess_for_naive_bayes(X_original)
X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb, y, test_size=0.2, stratify=y, random_state=42)
X_train_nb = preprocess_and_scale(X_train_nb)
X_test_nb = preprocess_and_scale(X_test_nb)

nb_model = GaussianNB(priors=[0.5, 0.5])
nb_model.fit(X_train_nb, y_train_nb)
y_pred_nb = nb_model.predict(X_test_nb)
results['Naive Bayes'] = {
    'accuracy': accuracy_score(y_test_nb, y_pred_nb),
    'auc': roc_auc_score(y_test_nb, nb_model.predict_proba(X_test_nb)[:, 1]),
    'classification_report': classification_report(y_test_nb, y_pred_nb),
    'confusion_matrix': confusion_matrix(y_test_nb, y_pred_nb)
}


# Model1: Decision Tree
X_dt = preprocess_for_tree(X_original)
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X_dt, y, test_size=0.2, stratify=y, random_state=42)

dt_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
dt_model.fit(X_train_dt, y_train_dt)
y_pred_dt = dt_model.predict(X_test_dt)
results['Decision Tree'] = {
    'accuracy': accuracy_score(y_test_dt, y_pred_dt),
    'auc': roc_auc_score(y_test_dt, dt_model.predict_proba(X_test_dt)[:, 1]),
    'classification_report': classification_report(y_test_dt, y_pred_dt),
    'confusion_matrix': confusion_matrix(y_test_dt, y_pred_dt)
}


# Model2: Logistic Regression
X_log = preprocess_for_logistic(X_original)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y, test_size=0.2, stratify=y, random_state=42)
X_train_log = preprocess_and_scale(X_train_log)
X_test_log = preprocess_and_scale(X_test_log)

log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train_log, y_train_log)
y_pred_log = log_model.predict(X_test_log)
results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test_log, y_pred_log),
    'auc': roc_auc_score(y_test_log, log_model.predict_proba(X_test_log)[:, 1]),
    'classification_report': classification_report(y_test_log, y_pred_log),
    'confusion_matrix': confusion_matrix(y_test_log, y_pred_log)
}


# Model3: LightGBM
X_lgb = preprocess_for_tree(X_original)
class_counts = y.value_counts()
scale_pos_weight = class_counts[0] / class_counts[1]  # Needed for LGBM

# Clean the column names before LightGBM
X_lgb.columns = X_lgb.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

X_train_lgb, X_test_lgb, y_train_lgb, y_test_lgb = train_test_split(X_lgb, y, test_size=0.2, stratify=y, random_state=42)
lgb_model = LGBMClassifier(scale_pos_weight=scale_pos_weight)
lgb_model.fit(X_train_lgb, y_train_lgb)
y_pred_lgb = lgb_model.predict(X_test_lgb)
results['LightGBM'] = {
    'accuracy': accuracy_score(y_test_lgb, y_pred_lgb),
    'auc': roc_auc_score(y_test_lgb, lgb_model.predict_proba(X_test_lgb)[:, 1]),
    'classification_report': classification_report(y_test_lgb, y_pred_lgb),
    'confusion_matrix': confusion_matrix(y_test_lgb, y_pred_lgb)
}


# Display Results
for name, result in results.items():
    print(f"\n {name} Results")
    print("Accuracy:", result['accuracy'])
    print("ROC AUC:", result['auc'])
    print("Confusion Matrix:\n", result['confusion_matrix'])
    print("Classification Report:\n", result['classification_report'])
    

#For clinical_notes
from sklearn.feature_extraction.text import TfidfVectorizer
# 1. Fill missing notes
df['clinical_notes'] = df['clinical_notes'].fillna("")

# 2. Convert to TF-IDF features
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')  # Adjust features for performance
X_text = vectorizer.fit_transform(df['clinical_notes'])

# 3. Target variable
y = df['diabetes']

# 4. Split
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
    X_text, y, test_size=0.2, stratify=y, random_state=42
)

text_model = LogisticRegression(max_iter=1000)
text_model.fit(X_train_text, y_train_text)

y_pred_text = text_model.predict(X_test_text)
y_probs_text = text_model.predict_proba(X_test_text)[:, 1]

print("\n Clinical Notes Only Results:")
print("Accuracy:", accuracy_score(y_test_text, y_pred_text))
print("AUC:", roc_auc_score(y_test_text, y_probs_text))
print(classification_report(y_test_text, y_pred_text))

#combining both
# Preprocess structured features
X_structured = df.drop(columns=['diabetes', 'clinical_notes'])
X_structured = pd.get_dummies(X_structured, drop_first=True)
X_structured = X_structured.fillna(0)

# Scale structured data

scaler = StandardScaler()
X_structured_scaled = scaler.fit_transform(X_structured)

# Convert structured to sparse matrix to match TF-IDF
from scipy.sparse import hstack

X_text_struct = hstack([X_text, X_structured_scaled])

# Train-test split
X_train_combo, X_test_combo, y_train_combo, y_test_combo = train_test_split(
    X_text_struct, y, test_size=0.2, stratify=y, random_state=42
)

# Train logistic regression on combined features
combo_model = LogisticRegression(max_iter=1000)
combo_model.fit(X_train_combo, y_train_combo)

y_pred_combo = combo_model.predict(X_test_combo)
y_probs_combo = combo_model.predict_proba(X_test_combo)[:, 1]

print("\n Clinical Notes + Structured Data Results:")
print("Accuracy:", accuracy_score(y_test_combo, y_pred_combo))
print("AUC:", roc_auc_score(y_test_combo, y_probs_combo))
print(classification_report(y_test_combo, y_pred_combo))


#result table

# Create a results summary dictionary
summary_data = []

for model_name, metrics in results.items():
    report = metrics['classification_report']
    if isinstance(report, dict):  # If stored as dict (from output_dict=True)
        precision = report['1']['precision']
        recall = report['1']['recall']
        f1_score = report['1']['f1-score']
    else:
        precision = recall = f1_score = None  # fallback in case it's not parsed

    summary_data.append({
        'Model': model_name,
        'Accuracy': round(metrics['accuracy'] * 100, 2),
        'AUC': round(metrics['auc'] * 100, 2),
        'Precision': round(precision * 100, 2) if precision else 'N/A',
        'Recall ': round(recall * 100, 2)if recall else 'N/A',
        'F1-Score ': round(f1_score * 100, 2)if f1_score else 'N/A' ,
    })

# Convert to DataFrame
results_df = pd.DataFrame(summary_data)

# Display as a table
print(results_df.to_string(index=False))

print(report)






'''missclassification, class imbalance'''

df['diabetes'].value_counts(normalize=True)
df['diabetes'].value_counts().plot.pie(autopct='%1.1f%%', labels=['No Diabetes', 'Diabetes'])
plt.title('Class Distribution (Before Undersampling)')
plt.ylabel('')
plt.show()


'''under sampling'''

from imblearn.under_sampling import RandomUnderSampler

from collections import Counter

# Let's say this is your training data:
# X_train_combo, y_train_combo

print("Before undersampling:", Counter(y_train_combo))

# Create the undersampler
rus = RandomUnderSampler(random_state=42)

# Apply undersampling
X_train_resampled, y_train_resampled = rus.fit_resample(X_train_combo, y_train_combo)

print("After undersampling:", Counter(y_train_resampled))

X_train_resampled, y_train_resampled = rus.fit_resample(X_train_combo, y_train_combo)



'''plot for undersampling'''
labels = ['No Diabetes', 'Diabetes']
sizes = [6800, 6800]
colors = ['#1f77b4', '#ff7f0e']

# Plot pie chart
plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
plt.axis('equal')
plt.title('Class Distribution (After Undersampling)', fontsize=14)
plt.show()



'''after'''
'''logistic'''
# Train a new logistic regression model on the undersampled data
undersampled_model = LogisticRegression(max_iter=1000)
undersampled_model.fit(X_train_resampled, y_train_resampled)

# Predict on the same original test set
y_pred_undersampled = undersampled_model.predict(X_test_combo)
y_probs_undersampled = undersampled_model.predict_proba(X_test_combo)[:, 1]


from sklearn.metrics import precision_score, recall_score, f1_score
'''decisiontree'''
dt_model_under = DecisionTreeClassifier(random_state=42)
dt_model_under.fit(X_train_resampled, y_train_resampled)

y_pred_dt_under = dt_model_under.predict(X_test_combo)
y_probs_dt_under = dt_model_under.predict_proba(X_test_combo)[:, 1]


'''lightgbm'''
lgb_model_under = LGBMClassifier()
lgb_model_under.fit(X_train_resampled, y_train_resampled)

y_pred_lgb_under = lgb_model_under.predict(X_test_combo)
y_probs_lgb_under = lgb_model_under.predict_proba(X_test_combo)[:, 1]


# Append to summary


# For Decision Tree
dt_under_metrics = {
    'Model': 'Decision Tree (Undersampled)',
    'Accuracy': round(accuracy_score(y_test_combo, y_pred_dt_under) * 100, 2),
    'AUC': round(roc_auc_score(y_test_combo, y_probs_dt_under) * 100, 2),
    'Precision': round(precision_score(y_test_combo, y_pred_dt_under) * 100, 2),
    'Recall ': round(recall_score(y_test_combo, y_pred_dt_under) * 100, 2),
    'F1-Score ': round(f1_score(y_test_combo, y_pred_dt_under) * 100, 2),
}

# For LightGBM
lgb_under_metrics = {
    'Model': 'LightGBM (Undersampled)',
    'Accuracy': round(accuracy_score(y_test_combo, y_pred_lgb_under) * 100, 2),
    'AUC': round(roc_auc_score(y_test_combo, y_probs_lgb_under) * 100, 2),
    'Precision': round(precision_score(y_test_combo, y_pred_lgb_under) * 100, 2),
    'Recall ': round(recall_score(y_test_combo, y_pred_lgb_under) * 100, 2),
    'F1-Score ': round(f1_score(y_test_combo, y_pred_lgb_under) * 100, 2),
}

# Add to results_df using pd.concat
results_df = pd.concat([results_df, pd.DataFrame([dt_under_metrics, lgb_under_metrics])], ignore_index=True)

undersampled_metrics = {
    'Model': 'LogReg (Undersampled)',
    'Accuracy': round(accuracy_score(y_test_combo, y_pred_undersampled) * 100, 2),
    'AUC': round(roc_auc_score(y_test_combo, y_probs_undersampled) * 100, 2),
    'Precision': round(precision_score(y_test_combo, y_pred_undersampled) * 100, 2),
    'Recall ': round(recall_score(y_test_combo, y_pred_undersampled) * 100, 2),
    'F1-Score ': round(f1_score(y_test_combo, y_pred_undersampled) * 100, 2),
}

results_df = pd.concat([results_df, pd.DataFrame([undersampled_metrics])], ignore_index=True)
print(results_df)




'''Feature importance'''
from sklearn.feature_selection import SelectKBest, f_classif
# Replace with your preprocessed dataset (e.g., encoded and scaled)
X = pd.get_dummies(X_structured, drop_first=True).fillna(0)
y = df['diabetes']

selector = SelectKBest(score_func=f_classif, k=10)  # Top 10 features
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
print("Top selected features:\n", selected_features)
# Apply SelectKBest
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Get scores and selected feature names
scores = selector.scores_
selected_mask = selector.get_support()
selected_scores = scores[selected_mask]
selected_features = X.columns[selected_mask]

# Create a DataFrame for plotting
score_df = pd.DataFrame({'Feature': selected_features, 'Score': selected_scores})
score_df = score_df.sort_values(by='Score', ascending=True)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Score', y='Feature', data=score_df, palette='viridis')
plt.title('Top 10 Features by ANOVA F-Score (SelectKBest)')
plt.xlabel('F-Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Use the model trained on the original or resampled data
lgb_model.fit(X_train_lgb, y_train_lgb)

# Get feature importance
importances = lgb_model.feature_importances_
features = X_lgb.columns

# Create a DataFrame
feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

# Plot top 10
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df.head(10), x='Importance', y='Feature', palette='viridis')
plt.title('Top 10 Feature Importances (LightGBM)')
plt.tight_layout()
plt.show()





# Fit model if not already trained
dt_model.fit(X_train_dt, y_train_dt)

# Get feature importance
importances = dt_model.feature_importances_
features = X_dt.columns

# Create and plot DataFrame
feat_imp_dt = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_imp_dt = feat_imp_dt.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_dt.head(10), x='Importance', y='Feature', palette='magma')
plt.title('Top 10 Feature Importances (Decision Tree)')
plt.tight_layout()
plt.show()

'''weka'''
# Save as CSV

# Convert sparse matrix to dense array (if needed)
X_dense = X_train_resampled.toarray() if hasattr(X_train_resampled, "toarray") else X_train_resampled

# Create DataFrame
df_resampled = pd.DataFrame(X_dense)

# Add the target column
df_resampled['Class'] = y_train_resampled.values

# Save to CSV
df_resampled.to_csv('undersampled_data.csv', index=False)


'''final plots'''

from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc


# 1. 📊 Bar Chart: Accuracy vs AUC
plt.figure(figsize=(10, 5))
x = np.arange(len(results_df))
bar_width = 0.35

plt.bar(x, results_df['Accuracy'], width=bar_width, label='Accuracy')
plt.bar(x + bar_width, results_df['AUC'], width=bar_width, label='AUC')
plt.xticks(x + bar_width / 2, results_df['Model'], rotation=45)
plt.ylabel('Score (%)')
plt.title('Model Accuracy vs AUC')
plt.legend()
plt.tight_layout()
plt.show()

# 2. 📈 ROC Curve Comparison
plt.figure(figsize=(8, 6))

# You must define a dictionary of models and test sets first
roc_models = {
    "Decision Tree": (dt_model, X_test_dt, y_test_dt),
    "Logistic Regression": (log_model, X_test_log, y_test_log),
    "LightGBM": (lgb_model, X_test_lgb, y_test_lgb),
}

for name, (model, X_test, y_test) in roc_models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. 📉 Confusion Matrix – LightGBM
ConfusionMatrixDisplay.from_estimator(
    lgb_model, X_test_lgb, y_test_lgb,
    display_labels=["No Diabetes", "Diabetes"],
    cmap=plt.cm.Blues,
    values_format='d'
)
plt.title("Confusion Matrix – LightGBM")
plt.tight_layout()
plt.show()




