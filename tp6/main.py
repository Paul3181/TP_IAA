import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./donnees/data2.csv')

X = df[df.columns[1:]]
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


