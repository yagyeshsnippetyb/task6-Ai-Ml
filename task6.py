import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = {
    'Player': ['Virat', 'Rohit', 'Dhoni', 'Rahul', 'Bumrah', 'Hardik', 'Ashwin', 'Shami', 'Gill', 'Pant'],
    'Runs': [12000, 11000, 10500, 6000, 500, 2500, 1000, 300, 4000, 2000],
    'Matches': [250, 230, 300, 120, 70, 100, 90, 80, 60, 75],
    'Is_Batsman': [1, 1, 1, 1, 0, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df[['Runs', 'Matches']]
y = df['Is_Batsman']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

plt.scatter(X['Runs'], X['Matches'], c=y, cmap='coolwarm')
plt.xlabel('Runs')
plt.ylabel('Matches')
plt.title('Cricketer Classification (Batsman/Non-Batsman)')
plt.show()
