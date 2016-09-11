import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn import svm


train_df = pd.read_csv('./csvs/train.csv', header=0)
train_df = train_df.dropna()
train_df.drop('Name', axis=1, inplace=True)
train_df.drop('Cabin', axis=1, inplace=True)
train_df.drop('Ticket', axis=1, inplace=True)
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
train_df['EmbarkedGroup'] = train_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train_df.drop('Sex', axis=1, inplace=True)
train_df.drop('Embarked', axis=1, inplace=True)
train_data = train_df[['Survived','PassengerId','Pclass','Age','SibSp','Parch','Fare','Gender','EmbarkedGroup']].values

test_df = pd.read_csv('./csvs/test.csv', header=0)
test_df = test_df.fillna(0)
test_df.drop('Name', axis=1, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)
test_df.drop('Ticket', axis=1, inplace=True)
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test_df['EmbarkedGroup'] = test_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_df.drop('Sex', axis=1, inplace=True)
test_df.drop('Embarked', axis=1, inplace=True)
test_data = test_df[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Gender','EmbarkedGroup']].values

result = { 'score': 1, 'num': 0, 'estimator': 0 }

for _x in range(1, 10):
  for num in range(1,5):
    estimator = svm.SVM()
    # forest = RandomForestClassifier(n_estimators = num)
    estimator = estimator.fit(train_data[0::,1::],train_data[0::,0])
    output = estimator.predict(test_data)
    print 'estimator'
    print num
    print 'train'
    print train_data[:,0].sum() / train_data[:,0].size
    train = train_data[:,0].sum() / train_data[:,0].size
    print 'test'
    print output.sum() / test_data[:,0].size
    test = output.sum() / test_data[:,0].size
    print '\n'
    if result['score'] > np.abs(train - test):
        result['score'] = np.abs(train - test)
        result['num'] = num
        result['estimator'] = estimator

print result

estimator = result['estimator']
output = estimator.predict(test_data)
data = { 'PassengerId': test_df['PassengerId'].values, 'Survived': output }
result_df = pd.DataFrame(data)
result_df.to_csv('./csvs/svm.csv', index=False)
