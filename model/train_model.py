#Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os 

#To set the main input data file path 
csv_path=os.getenv("DATASET_PATH","model/creditcard.csv")

#Load the data and return the dataframe
def load_data(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df=pd.read_csv(path)
    return df

def preprocess_data(df):
    """
    Scales "Amount" column
    Remove "Time" columnn
    Return X (input features) and y(label(fraud/not fraud))
    """
    df=df.copy()
    df.drop("Time",axis=1,inplace=True)

    scaler=StandardScaler()
    df["Amount"]=scaler.fit_transform(df["Amount"])
    X=df.drop("Class",axis=1)
    y=df["Class"]
    return X,y
def balance_classes(X,y):
    """
        To balance normal ratio to fraud ratio in dataset 
    """
    fraud_indices=y[y==1].index
    normal_indices=y[y==0].index
    normal_sample=normal_indices[:len(fraud_indices)]
    balanced_indices=fraud_indices.union(normal_sample)
    X_balanced=X.loc[balanced_indices]
    y_balanced=y.loc[balanced_indices]
    return X_balanced,y_balanced

def split_data(X,y,test_size=0.3,random_state=42):
    #To split the data in to training set and testing set 
    return train_test_split(X,y,test_size=test_size,stratify=y,random_state=random_state)

def train_model(X_train, y_train):
    """
    Trains a Random Forest classifier.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and prints key metrics.
    """
    y_pred = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))


def save_model(model,filename="model/fraud_model.pkl"):
    #Save the train model to a file
    os.makedirs(os.path.dirname(filename),exist_ok=True)
    joblib.dump(model,filename)
    print(f"Model saved to {filename}")




if __name__=="__main__":
    df=load_data(csv_path)
    X,y=preprocess_data(df)
    X_bal,y_bal=balance_classes(X,y)
    X_train,X_test,y_train,y_test=split_data(X_bal,y_bal)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)



    



