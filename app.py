import pickle
from flask import Flask,request,app,jsonify,url_for,render_template,redirect
import numpy as np
import pandas as pd
from flask import flash

# To solve session error
from flask import session


app=Flask(__name__)
SESSION_TYPE = 'memcache'


# Loading the model
reg_model=pickle.load(open("diabities_log_reg.pkl",'rb'))
scaler=pickle.load(open("Scaler.pkl",'rb'))


@app.route("/") # -----> / implies blank url
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
        # return jsonify(404)
        pass

@app.route("/predict_result",methods=['POST'])
def predict_result():
    try:
        print(list(request.form.values()))
        data=[float(x) for x in list(request.form.values())]
        # print(data)
        data=[data]
        # print(data)
        data[0][1:6]=scaler.transform(np.array([data[0][1:6]])).flatten()
        ans=reg_model.predict(data)[0]
        if(ans==1):
            flash(" The person is prone to Diabetic ",'danger') # flash(Message,category)
        else:
            flash(" The person not Diabetic ",'success') # flash(Message,category)
        return redirect('/')
    except:
        return render_template("Error_page.html")


if(__name__=="__main__"):


    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.run(debug=True)