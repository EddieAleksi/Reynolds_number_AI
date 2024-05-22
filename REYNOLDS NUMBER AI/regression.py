from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split[(X, y, test_size=0.4, random_state =0)]

model_logistic = LogisticRegression()
model_logistic.fit(X_train, y_train)
y_pred_,logistic = model_logistic.predict(X_test)

print("Training set score: {:2f}".format(model_logistic.score(X_train, y_train)))
print("Test set score: {:.2f}".format(model_logistic.score)