from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)


@app.route('/',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        model=pickle.load(open('diabetes.sav','rb'))
        hbp=float(request.form['bp'])
        highchol=float(request.form['ch'])
        cholcheck=float(request.form['cholcheck'])
        bmi=float(request.form['bmi'])
        cigarettes=float(request.form['cig'])
        stroke=float(request.form['stk'])
        chd=float(request.form['heartdisease'])
        phy_act=float(request.form['phy_act'])
        fruit=float(request.form['fruit_cons'])
        veg=float(request.form['veg_cons'])
        alcohol=float(request.form['alcohol_cons'])
        health_coverage=float(request.form['hcc'])
        doccost=float(request.form['cost'])
        health=float(request.form['gen_hel'])
        poor_mental_health=float(request.form['num_ment_hlth'])
        physical_illnes=float(request.form['num_phy_hlth'])
        diff_walking=float(request.form['walk'])
        gender=float(request.form['gender'])
        age=float(request.form["age"])
        education=float(request.form['education'])
        income=float(request.form['income'])

    
        input_data=np.array([hbp,highchol,cholcheck,bmi,cigarettes,stroke,chd,
        phy_act,fruit,veg,alcohol,health_coverage,doccost,health,poor_mental_health,physical_illnes,diff_walking,gender,age,education,income])
        input_data=input_data.reshape(-1,21)
        predictions=model.predict_proba(input_data)
        pred_class=model.predict(input_data)
        print(pred_class)
        # index=list(predictions[0]).index(max(predictions[0]))
        pred_class=int(pred_class)
        if pred_class==0:
            msg="You don't have diabetes"
        elif pred_class==1:
            msg="You have "+str(predictions[0][pred_class]*100)+"% chances of having prediabetes"
        else:
            msg="You have "+str(predictions[0][pred_class]*100)+"% chances of having diabetes"
        return render_template('index.html',result=msg)
        
    else:
        return render_template('index.html')
    



if __name__=="__main__":
    app.run(debug=False)