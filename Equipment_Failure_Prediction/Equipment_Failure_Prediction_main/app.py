from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

app = Flask(__name__)

# Load the trained logistic regression model
model = pickle.load(open('model.pkl', 'rb'))

# Load the dataset containing equipment health information
data = pd.read_csv("iot_sensor_dataset.csv")

# Preprocess the data if necessary

# Calculate probabilities of equipment failures
data['Probability'] = model.predict_proba(data.iloc[:, :-1])[:, 1]

# Create a Pie Chart
fig = px.pie(data, values='Probability', names=pd.cut(data['Probability'], bins=[0, 0.25, 0.5, 0.75, 1], labels=['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1']))

@app.route('/')
def hello_world():
    return render_template("equip_pred.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        int_features = [int(x) for x in request.form.values()]
        final = [np.array(int_features)]
        prediction = model.predict_proba(final)
        output = '{0:.{1}f}'.format(prediction[0][1], 2)

        if output > str(0.65):
            return render_template('equip_pred.html', pred='Your Equipment is Prone to Failure.\nProbability of Equipment Failure is {}'.format(output))
        else:
            return render_template('equip_pred.html', pred='Your Equipment is not Prone to Failure.\n Probability of Equipment Failure is {}'.format(output))
    else:
        return render_template('equip_pred.html')

@app.route('/dashboard')
def dashboard():
    # Filter equipment with probability of failure between 0.75 and 1
    high_probability_equipment = data[(data['Probability'] >= 0.75) & (data['Probability'] <= 1)][['Probability']]
    
    # Add index as Equipment ID
    high_probability_equipment['Equipment ID'] = high_probability_equipment.index
    
    # Render template with pie chart and table
    return render_template("dashboard.html", pie_chart=fig.to_html(), high_probability_equipment=high_probability_equipment.to_html(index=False))

if __name__ == '__main__':
    app.run(debug=True)
