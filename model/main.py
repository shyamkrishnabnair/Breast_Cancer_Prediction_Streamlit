import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle as pkl


def get_clean_data():
    #laod the data
    data=pd.read_csv('data/data.csv')

    #clean the data 
    # data = data.dropna()  # Example cleaning step: drop rows with missing values
    # data = data.reset_index(drop=True)  # Reset index after dropping rows
    data=data.drop(['Unnamed: 32','id'],axis=1)  # Drop specific columns
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})  # Map diagnosis to binary values

    return data


def create_model(data):
    X=data.drop('diagnosis',axis=1)  # Features
    Y=data['diagnosis']  # Target variable

    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)  # Scale the features

    #train-test split
    X_train,X_test,Y_train,Y_test=train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

    #create the model
    model=LogisticRegression()
    model.fit(X_train, Y_train)  # Train the model

    # accuracy_1=accuracy_score(Y_train, model.predict(X_train))  # Calculate accuracy on training set
    # accuracy_2=accuracy_score(Y_test, model.predict(X_test))  # Calculate accuracy on testing set

    # print(f"Model accuracy on training set: {accuracy_1:.2f}")
    # print(f"Model accuracy on testing set: {accuracy_2:.2f}")

    return model,scaler

def main():
    data = get_clean_data()
    # print(data.head())  # Display the first few rows of the cleaned data
    # print(data.info())  # Display information about the DataFrame

    model,scaler = create_model(data)

    with open('model/model.pkl','wb') as f:
        pkl.dump(model,f)

    with open('model/scaler.pkl','wb') as f:
        pkl.dump(scaler,f)


if __name__ == "__main__":
    main()