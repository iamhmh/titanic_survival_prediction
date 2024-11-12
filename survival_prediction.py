import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_data():
    train_data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
    return train_data

def clean_data(df):
    df_clean = df.copy()
    
    df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
    df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
    
    df_clean['Sex'] = df_clean['Sex'].map({'female': 1, 'male': 0})
    embarked_dummies = pd.get_dummies(df_clean['Embarked'], prefix='Embarked')
    df_clean = pd.concat([df_clean, embarked_dummies], axis=1)
    
    df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1
    df_clean['IsAlone'] = (df_clean['FamilySize'] == 1).astype(int)
    
    df_clean['Title'] = df_clean['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Dr": "Other",
        "Rev": "Other",
        "Col": "Other",
        "Major": "Other",
        "Mlle": "Miss",
        "Countess": "Other",
        "Ms": "Miss",
        "Lady": "Other",
        "Jonkheer": "Other",
        "Don": "Other",
        "Dona": "Other",
        "Mme": "Mrs",
        "Capt": "Other",
        "Sir": "Other"
    }
    df_clean['Title'] = df_clean['Title'].map(title_mapping)
    title_dummies = pd.get_dummies(df_clean['Title'], prefix='Title')
    df_clean = pd.concat([df_clean, title_dummies], axis=1)
    
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone'] + \
               list(embarked_dummies.columns) + list(title_dummies.columns)
    
    return df_clean[features], df_clean['Survived']

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("\nClassification Report :")
    print(classification_report(y_test, predictions))
    
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Values')
    plt.xlabel('Predictions')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Features importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def main():
    print("Loading data...")
    data = load_data()
    
    print("Preparing data...")
    X, y = clean_data(data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    numerical_features = ['Age', 'Fare', 'FamilySize']
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    
    print("Training the model...")
    model = train_model(X_train, y_train)
    
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()