import pandas as pd
from flask import Flask, render_template, request
import json
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
from sklearn.metrics import accuracy_score,f1_score
import plotly.figure_factory as ff


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



@app.route("/logistic_regression")
def logistic_regression():
    #logistic regression
    logreg = LogisticRegression(random_state=16)
    logreg.fit(X_train_scaled, y_train.values.ravel())
    y_pred = logreg.predict(X_test_scaled)

    #confusion matrix
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    #accuracy, f1 score, 
    acc_log = metrics.accuracy_score(y_test, y_pred)
    f1_log = metrics.f1_score(y_test, y_pred)

    ##set up annotation of confusion matrix
    x = ['no diabetes','diabetes']
    y = ['no diabetes','diabetes']
    z_text = [[str(y) for y in x] for x in cnf_matrix]

    fig = ff.create_annotated_heatmap(cnf_matrix, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # add title
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                    #xaxis = dict(title='x'),
                    #yaxis = dict(title='x')
                    )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))
    
    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                        x=-0.35,
                        y=0.5,
                        showarrow=False,
                        text="Real value",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))
    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    header="Analysis Results of Logistic Regression"
    description = f"""
    F1 score is {f1_log}. \n Accuracy is {acc_log}. 
    """
    return render_template('plotly.html', graphJSON=graphJSON, header=header,description=description)

@app.route('/rf/')
def rf():
    return render_template('rf.html')

@app.route('/rf/info',methods=["GET"])
def rf_graph():
    num = request.args.get("number of trees")
    num = int(num)
    clf=RandomForestClassifier(n_estimators=num)
    clf.fit(X_train_scaled,y_train)
    y_pred2=clf.predict(X_test_scaled)
    cnf_matrix2 = metrics.confusion_matrix(y_test, y_pred2)
    
    #accuracy, f1 score, 
    acc_rf = metrics.accuracy_score(y_test, y_pred2)
    f1_rf = metrics.f1_score(y_test, y_pred2)

    ##set up annotation of confusion matrix
    x = ['no diabetes','diabetes']
    y = ['no diabetes','diabetes']
    z_text = [[str(y) for y in x] for x in cnf_matrix2]

    fig2 = ff.create_annotated_heatmap(cnf_matrix2, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # add title
    fig2.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                    #xaxis = dict(title='x'),
                    #yaxis = dict(title='x')
                    )

    # add custom xaxis title
    fig2.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))
    
    # add custom yaxis title
    fig2.add_annotation(dict(font=dict(color="black",size=14),
                        x=-0.35,
                        y=0.5,
                        showarrow=False,
                        text="Real value",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))
    # adjust margins to make room for yaxis title
    fig2.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig2['data'][0]['showscale'] = True

    graphJSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    header="Analysis Results of Random Forest"
    description = f"""
    F1 score is {f1_rf}. \n Accuracy is {acc_rf}. 
    """
    return render_template('plotly.html', graphJSON=graphJSON, header=header,description=description)

@app.route('/knn/')
def knn():
    return render_template('knn.html')


@app.route('/knn/info', methods=["GET"])
def knn_graph():

    num = request.args.get("k")
    num = int(num)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled,y_train)
    y_pred3 = knn.predict(X_test_scaled)
    cnf_matrix3 = metrics.confusion_matrix(y_test, y_pred3)
    cnf_matrix3
   
    #accuracy, f1 score, 
    acc_knn = metrics.accuracy_score(y_test, y_pred3)
    f1_knn = metrics.f1_score(y_test, y_pred3)

    ##set up annotation of confusion matrix
    x = ['no diabetes','diabetes']
    y = ['no diabetes','diabetes']
    z_text = [[str(y) for y in x] for x in cnf_matrix3]

    fig3 = ff.create_annotated_heatmap(cnf_matrix3, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # add title
    fig3.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                    #xaxis = dict(title='x'),
                    #yaxis = dict(title='x')
                    )

    # add custom xaxis title
    fig3.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))
    
    # add custom yaxis title
    fig3.add_annotation(dict(font=dict(color="black",size=14),
                        x=-0.35,
                        y=0.5,
                        showarrow=False,
                        text="Real value",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))
    # adjust margins to make room for yaxis title
    fig3.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig3['data'][0]['showscale'] = True

    graphJSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    header="Analysis Results of Logistic Regression"
    description = f"""
    F1 score is {f1_knn}. \n Accuracy is {acc_knn}. 
    """
    return render_template('plotly.html', graphJSON=graphJSON, header=header,description=description)
if __name__ == "__main__":
    app.run(debug=True)