# Import libraries
import pandas  as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
import pickle
import numpy as np
from flask import Flask, request, jsonify,render_template
import pickle
from last import abc

app = Flask(__name__)
# Load the model


@app.route("/")
def home():
    return render_template('home.html', h_beat="heartbeat-min.png")

@app.route('/api',methods=['POST'])
def predict_value():
    # Get the data from the POST request.
    form_data = list(request.form.values())
    #form_data=form_data[:-1]
    chukar = abc(form_data)

    if chukar == [2]:
    	msg = "Sorry, the person won't Survive"
    	alert = "danger"
    	beat = "dead.jpg"
    else:
    	msg = "Congrats, the person will be safe and sound"
    	alert = "success"
    	beat = "heartbeat-min.png"

    return render_template('home.html',msg=msg,  h_beat=beat, alert=alert)
if __name__ == '__main__':
    app.run(port=8000,debug=True)
    