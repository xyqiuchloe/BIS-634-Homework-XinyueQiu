import pandas as pd
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import plotly.express as px
import plotly
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,plot_confusion_matrix


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

data = pd.read_csv('health_data.csv')
#Train test split
y_cols = ['Diabetes']
X = data.drop(y_cols,axis=1)
y = data[y_cols]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10,stratify=y)

#data preprocessing: standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#logistic regression
logreg = LogisticRegression(random_state=16)
logreg.fit(X_train_scaled, y_train.values.ravel())
y_pred = logreg.predict(X_test_scaled)

#confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
#accuracy, f1 score, 
acc_log = metrics.accuracy_score(y_test, y_pred)

@app.route("/logistic_regression")
def log_reg_func(name):
    df = pd.read_csv("incd.csv")
    rate = df[df["State"] == name]["Rate"].item()
    return {"name": name, "age-adjusted_incidence_rate": rate}
    
@app.route("/info", methods=["GET"])
def info():
    name = request.args.get('name')
    df = pd.read_csv("incd.csv")
    if name in df["State"].values:
        rate = df[df["State"] == name]["Rate"].item()
    else:
        rate = "none"
    return render_template("info.html", name = name, rate = rate)

if __name__ == "__main__":
    app.run(debug=True)