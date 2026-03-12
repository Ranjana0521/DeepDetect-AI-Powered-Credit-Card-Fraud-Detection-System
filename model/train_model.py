import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class",axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(
X_scaled,y,test_size=0.2,random_state=42
)

model = XGBClassifier(
n_estimators=200,
max_depth=6
)

model.fit(X_train,y_train)

joblib.dump(model,"fraud_model.pkl")
joblib.dump(scaler,"scaler.pkl")