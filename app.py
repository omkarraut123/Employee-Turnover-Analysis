import numpy as np
#import pandas as pd
import jsonify
from flask import Flask, request, jsonify, render_template
import pickle
import sys


app = Flask(__name__)
model = pickle.load(open('random_forest_tree_model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    satisfaction_level=float(request.form['satisfaction_level'])
    last_evaluation=float(request.form['last_evaluation'])
    number_project=float(request.form['number_project'])
    average_montly_hours=float(request.form['average_montly_hours'])
    time_spend_company=float(request.form['time_spend_company'])
    Work_accident=float(request.form['Work_accident'])
    promotion_last_5years=float(request.form['promotion_last_5years'])
    department=str(request.form['department'])
    salary=str(request.form['salary'])
    
    department_map = {"sales": 0, "accounting": 1, "hr": 2, "technical": 3, "support": 4,"management": 5, "IT": 6, "product_mng": 7, "marketing": 8, "RandD": 9}
    salary_map = {"low": 0, "medium": 1, "high": 2}
    list1=[]
    list1.append(satisfaction_level)
    list1.append(last_evaluation)
    list1.append(number_project)
    list1.append(average_montly_hours)
    list1.append(time_spend_company)
    list1.append(Work_accident)
    list1.append(promotion_last_5years)
    list1.append(float(department_map[department]))
    list1.append(float(salary_map[salary]))
    
    ind = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'department', 'salary']
    # test_x1 = pd.DataFrame(list1, index = ind)
    
    final_features = [np.array(list1)]
    prediction = model.predict(final_features)
    if(prediction == 0):
        result = 'Employee will not leave Organization!'
    else:
        result = 'Employee will be leave Organization! '

    output = result

    return render_template('index.html', prediction_text='Result :  {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)