import pickle
from flask import Flask,request,app,jsonify,url_for,render_template,redirect
import numpy as np
import pandas as pd


app=Flask(__name__)

# Loading the model
reg_model=pickle.load(open("diabities_log_reg.pkl",'rb'))
scaler=pickle.load(open("Scaler.pkl",'rb'))


@app.route("/") # -----> / implies blank url
def test():
    return render_template("index.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/about")
def main():
    return render_template("about.html")

@app.route("/contact_mail")
def contact_email():
    return render_template("contact_email.html")

@app.route("/error")
def error_found():
    return render_template("Error_page.html")

@app.route("/predict_result",methods=['POST'])
def predict_res():
    try:
        data=request.json['data']
        print(data)
        x=[list(data.values())]
        x[0][1:6]=scaler.transform(np.array([x[0][1:6]])).flatten()
        ans=reg_model.predict(x)[0]
        print(ans)
        return jsonify(int(ans))
    except:
        return render_template("Error_page.html")


if(__name__=="__main__"):
    app.run(debug=True)