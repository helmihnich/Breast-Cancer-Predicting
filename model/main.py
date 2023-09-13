import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle5 as pickle


def get_clean_data():
    data= pd.read_csv('./data/data.csv')
    data = data.drop(["Unnamed: 32","id"], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
    print(data.head())                              

    return data


def create_model(data): 
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    #scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #split the data
    x_train , x_test, y_train, y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )


    #train
    model = LogisticRegression()
    model.fit(x_train, y_train)


    #test
    y_pred = model.predict(x_test)
    print(f'Accuracy of our model : {accuracy_score(y_test, y_pred)} ')
    print(f'Classification report : {classification_report(y_test, y_pred)}')
    return model, scaler







def main():
    #Clean Data
    data = get_clean_data()


    #Creation the model
    model, scaler = create_model(data)

    #Export the model
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()