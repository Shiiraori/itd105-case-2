import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats.mstats import winsorize


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

removecol = 'time'
df = df.drop(removecol, axis=1)


df.info()
# winzorcol = ['serum_creatinine', 'platelets', 'creatinine_phosphokinase']
df['serum_creatinine'] = winsorize(df['serum_creatinine'], limits=[0.08, 0.08])
df['creatinine_phosphokinase'] = winsorize(df['creatinine_phosphokinase'], limits=[0.03, 0.03])
df['platelets'] = winsorize(df['platelets'], limits=[0.03, 0.03])


# Split the data into features (X) and target variable (y)
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=20, random_state=10)



model = LogisticRegression(max_iter=600, solver='lbfgs', C=1.5)
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

joblib.dump(model, 'model.pkl')


