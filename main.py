
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

x = np.array([1,2,0,2,0,1,1,2,0,0,0,1,1,2,1,2,0,2,0,1,1,2,0,0,0,1,1,2,1,2,0,2,0,1,1,2,0,0,0,1,1,2,1,2,0,2,0,1,1,2,0,0,0,1,1,2,1,2,0,2,0,1,1,2,0,0,0,1,1,2,1,2,0,2,0,1,1,2,0,0,0,1,1,2])
x=x.reshape(-1,1)

y = np.array([2,0,1,0,1,2,2,0,1,1,1,2,2,0,2,0,1,0,1,2,2,0,1,1,1,2,2,0,2,0,1,0,1,2,2,0,1,1,1,2,2,0,2,0,1,0,1,2,2,0,1,1,1,2,2,0,2,0,1,0,1,2,2,0,1,1,1,2,2,0,2,0,1,0,1,2,2,0,1,1,1,2,2,0,])

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {100*accuracy}%")

predictions = model.predict([[0]])
moves = {0:'R',1:'P',2:'S'}

print(f"Prediction: {moves[predictions[0]]}")