import os 
import pandas as pd
import argparse
import torch
from sklearn.linear_model import LogisticRegression
# !pip install sentence_transformers
from sentence_transformers import SentenceTransformer


def load_data():
    print("Loading data...")
    train_data = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    print("Load data done!")
    return train_data, test_data

def encode(train, test, device):
    print("Encoding..")
    # Load pre-trained model
    senttrans_model = SentenceTransformer('all-MiniLM-L6-v2',device=device)

    # Create embeddings for training set text
    X_train = train['Text'].values.tolist()
    X_train = [senttrans_model.encode(doc) for doc in X_train]

    # Create embeddings for test set text
    X_test = test['Text'].values.tolist()
    X_test = [senttrans_model.encode(doc) for doc in X_test]
    print("Encode done!")
    return X_train, X_test

def main():
    # ##Load Data
    train_data, test_data = load_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_test = encode(train_data, test_data, device)

    # ## Train classification model

    # Train a classification model using logistic regression classifier
    print("Training model...")
    y_train = train_data['sentiment']
    logreg_model = LogisticRegression(solver='saga')
    logreg_model.fit(X_train,y_train)
    print("Model trained!")
    preds = logreg_model.predict(X_train)
    acc = sum(preds==y_train)/len(y_train)
    print('Accuracy on the training set is {:.3f}'.format(acc))

    # ## Evaluate model performance

    # Evaluate performance on the test set
    print("Evaluating model...")
    y_test = test_data['sentiment']
    preds = logreg_model.predict(X_test)
    acc = sum(preds==y_test)/len(y_test)
    print("Evluation done!")
    print('Accuracy on the test set is {:.3f}'.format(acc))

    # ## Save results
    train_data['sentiment_pred'] = logreg_model.predict(X_train)
    test_data['sentiment_pred'] = logreg_model.predict(X_test)


    train_data.to_csv(os.path.join(DATA_PATH, "transformer_train_bins.csv"))
    test_data.to_csv(os.path.join(DATA_PATH, "transformer_test_bins.csv"))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_path', type = str, default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    arg = parse_args()
    if arg.code_path == None:
        CODE_PATH = os.getcwd()
    else:
        CODE_PATH = arg.code_path
    BASE_PATH = os.path.abspath(os.path.join(CODE_PATH, ".."))
    DATA_PATH = os.path.join(BASE_PATH, "Data")
    MODEL_PATH = os.path.join(BASE_PATH, "Model")
    os.chdir(BASE_PATH)
    
    main()