from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('model_jupyter_loan_1.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    income = request.form.get('person_income')
    ownership = request.form.get('person_home_ownership')
    emp = request.form.get('person_emp_length')
    l_grade = request.form.get('loan_grade')
    l_amount = request.form.get('loan_amnt')
    int_rate = request.form.get('loan_int_rate')
    per_income = request.form.get('loan_percent_income')
    defaulter = request.form.get('cb_person_default_on_file')

    #result =  {'person_income':income}

    input_query = np.array([[income,ownership,emp,l_grade,l_amount,int_rate,per_income,defaulter]], dtype=object)

    result = model.predict(input_query)

    return jsonify({'Defaulter':str(result)})

    #return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)