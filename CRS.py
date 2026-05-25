from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

  
# fetch dataset 
statlog_german_credit_data = fetch_ucirepo(id=144) 

# data (as pandas dataframes) 
X = statlog_german_credit_data.data.features 
y = np.ravel(statlog_german_credit_data.data.targets)  

categorical_cols = list(X.select_dtypes(exclude=['number']).columns)
numerical_cols = list(X.select_dtypes(include=['number']).columns)

pipeline = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

pipeline = ImbPipeline(
    steps=[
        ('preprocessor', pipeline),
        ('smote', SMOTE(random_state=42)),
        ('LR', LogisticRegression())
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
score = accuracy_score(y_test, predictions)

print(f'Accuracy: {score:.2f}')
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))