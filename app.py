import pickle
from flask import Flask,request,app,jsonify,url_for,render_template,redirect
import numpy as np
import pandas as pd
from flask import flash

from flask import session


app=Flask(__name__)
SESSION_TYPE = 'memcache'


reg_model=pickle.load(open("diabities_log_reg.pkl",'rb'))
scaler=pickle.load(open("Scaler.pkl",'rb'))


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact_mail")
def contact_email():
    return render_template("contact_email.html")


@app.route("/error")
def error_found():
    return render_template("Error_page.html")


@app.route("/predict_api",methods=['POST'])
def predict_api():
    try:
        data=request.json['data']
        print(data)
        x=[list(data.values())]
        x[0][1:6]=scaler.transform(np.array([x[0][1:6]])).flatten()
        ans=reg_model.predict(x)[0]
        print(ans)
        return jsonify(int(ans))
    except:
        pass

@app.route("/predict_result",methods=['POST'])
def predict_result():
    try:
        data=[float(x) for x in list(request.form.values())]
        data=[data]
        data[0][1:6]=scaler.transform(np.array([data[0][1:6]])).flatten()
        ans=reg_model.predict(data)[0]
        if(ans==1):
            flash(" The person is prone to Diabetic ",'danger')
        else:
            flash(" The person not Diabetic ",'success')
        return redirect('/')
    except:
        return render_template("Error_page.html")


if(__name__=="__main__"):


    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.run(debug=True)