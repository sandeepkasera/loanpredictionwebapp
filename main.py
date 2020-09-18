import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle


import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

#import pickle





app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/predict',methods=['POST'])
def predict():
    df=pd.read_csv("train.csv")
    X=df.drop(["Loan_ID","Loan_Status"],axis=1)
    y=df["Loan_Status"]
    X["Gender"].fillna("Male",inplace=True)
    X["Married"].fillna("Yes",inplace=True)
    X["Dependents"].fillna("0",inplace=True)
    X["Self_Employed"].fillna("No",inplace=True)
    X["LoanAmount"].fillna(X["LoanAmount"].mean(),inplace=True)
    X['Loan_Amount_Term'].fillna(X['Loan_Amount_Term'].mean(),inplace=True)
    X['Credit_History'].fillna(X['Credit_History'].mean(),inplace=True)
    X=pd.get_dummies(X)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=30)
    nb=GaussianNB()
    nb.fit(X_train,y_train)

	
	
    gender = request.form["gender"]
    married = request.form["Marital_status"]
    dependents = request.form["dependents"]
    SelfEmployed = request.form["SelfEmployed"]
    Applicantincome = int(request.form["Applicantincome"])
    coapplicantincome = int(request.form["coapplicantincome"])
    loanamount = int(request.form["loanamount"])
    loanamountterm = int(request.form["loanamountterm"])
    credithistory = int(request.form["credithistory"])
    Education = request.form["education"]
    propertyarea = request.form["propertyarea"]
    data = [[gender,married,dependents,Education,SelfEmployed,Applicantincome,coapplicantincome,loanamount,loanamountterm,credithistory,propertyarea]]
    newdf = pd.DataFrame(data, columns = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area'])
    newdf=pd.get_dummies(newdf)
    missing_col=set(X_train.columns)-set(newdf.columns)
    for c in missing_col:
        newdf[c]=0
    newdf=newdf[X_train.columns]	
    yp=nb.predict(newdf)
    if (yp[0]=='Y'):
        a="Your Loan is approved, Please contact at HDFC Bank Any Branch for further processing"
    else:
        a ="Sorry ! Your Loan is not approved"
       
   
   

    return render_template('index.html', prediction_text='{}'.format(a),ReCheck="Re-Check")



if __name__ == "__main__":
    app.run(debug=True)