from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np

app=Flask(__name__)

model=pickle.load(open("models.pkl","rb"))
scalar=pickle.load(open("scalar.pkl","rb"))
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    # data=request.get_json()
    # print("predict finction")
    cgpa=float(request.form['cgpa'])
    iq=float(request.form['iq'])
    
    data=scalar.transform([[cgpa,iq]])
    prediction=model.predict(data)
    print("cgpa:",cgpa)
    print("iq:",iq)
    print("prediction:",prediction)
    
    if prediction[0]==1:
        result="Congratulation you have Placed"
        
    else:
        result="Sorry! you have to do hard-work"
    return render_template(
        "index.html",
        result=result,
        cgpa=cgpa,
        iq=iq
    )
if __name__=="__main__":
    app.run(debug=True)