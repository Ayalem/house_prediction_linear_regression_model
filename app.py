from flask import Flask,request,render_template
import joblib
import numpy as np

app=Flask(__name__)
best_model=joblib.load('house_price_model.pkl')
best_poly=joblib.load('polynomial_features.pkl')
best_scaler=joblib.load('scaler.pkl')
def predict_price(features):
     features_poly=best_poly.transform(features)
     features_poly_scaled=best_scaler.transform(features_poly)
     prediction=best_model.predict(features_poly_scaled)
     return prediction[0]

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods=['GET','POST'])
def predictions():
    if request.method=='GET':
        return render_template("predict.html")
    elif request.method=='POST':
         
     features=[float(request.form[key]) for key in request.form]
     features=[np.array(features)]
     prediction=predict_price(features)
     return render_template("predict.html",prediction=prediction.round(3))



if  __name__=='__main__':
        app.run(debug=True)