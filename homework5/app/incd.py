import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/state/<string:name>")
def state_api(name):
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