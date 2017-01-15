import pandas as pd
from sklearn.ensemble import RandomForestClassifier


#Simple method to filter a list of unwanted columns
def filter_unwanted_columns(train_data, test_data, unwanted_columns):
    for column in unwanted_columns:
        if (column in train_data) and (column in test_data):
            print('Deleting column: ',column )
            train_data = train_data.drop(column, axis = 1)
            test_data = test_data.drop(column, axis = 1)
    return train_data, test_data




if __name__ == '__main__':
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    unwanted_columns = ['Name', 'Ticket', 'Cabin', 'Embarked']

    train_data, test_data = filter_unwanted_columns(train_data, test_data, unwanted_columns)

    #Fill NaN data and replace gender with numbers for test_data
    test_data = test_data.fillna(method = "ffill")
    test_data.Sex.replace(to_replace = dict(female=1, male = 0), inplace = True)

    #Fill NaN data and replace gender with numbers for train_data
    train_data = train_data.fillna(method = 'ffill')
    train_data.Sex.replace(to_replace = dict(female=1, male = 0), inplace = True)
    survived = train_data.pop('Survived')


    #Just using a basic RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 100)
    forest = forest.fit(train_data, survived)
    res = forest.predict(test_data)
    df = pd.DataFrame(res)


    #This is only to write the results to a csv
    #df.to_csv('solution.csv')
