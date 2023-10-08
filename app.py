import os
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import pickle
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd

app = Flask(__name__)

disease_info = pd.read_csv('C:\\Users\\bayya\\OneDrive\\Desktop\\saikiranbls\\disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('C:\\Users\\bayya\\OneDrive\\Desktop\\saikiranbls\\supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("C:\\Users\\bayya\\OneDrive\\Desktop\\saikiranbls\\plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))
    



mod = pickle.load(open('C:\\Users\\bayya\\OneDrive\\Desktop\\saikiranbls\\model.pkl', 'rb'))
sc = pickle.load(open('C:\\Users\\bayya\\OneDrive\\Desktop\\saikiranbls\\standscaler.pkl', 'rb'))
ms = pickle.load(open('C:\\Users\\bayya\\OneDrive\\Desktop\\saikiranbls\\minmaxscaler.pkl', 'rb'))

mode = pickle.load(open('C:\\Users\\bayya\\OneDrive\\Desktop\\saikiranbls\\classifier.pkl', 'rb'))
ferti = pickle.load(open('C:\\Users\\bayya\\OneDrive\\Desktop\\saikiranbls\\fertilizer.pkl', 'rb'))

@app.route('/crop', methods=['GET', 'POST'])
def crop():
    result = None

    if request.method == 'POST':
        try:
            # Extract form data
            N = float(request.form['Nitrogen'])
            P = float(request.form['Phosporus'])
            K = float(request.form['Potassium'])
            temp = float(request.form['Temperature'])
            humidity = float(request.form['Humidity'])
            ph = float(request.form['Ph'])
            rainfall = float(request.form['Rainfall'])

            # Create a feature list
            feature_list = [N, P, K, temp, humidity, ph, rainfall]
            single_pred = np.array(feature_list).reshape(1, -1)

            # Scale the features
            scaled_features = ms.transform(single_pred)
            final_features = sc.transform(scaled_features)

            # Make prediction
            prediction = mod.predict(final_features)

            # Crop dictionary
            crop_dict = {
                1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
            }

            if prediction[0] in crop_dict:
                crop = crop_dict[prediction[0]]
                result = "{} is the best crop to be cultivated right there".format(crop)
            else:
                result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        except Exception as e:
            result = "Error: {}".format(str(e))

    return render_template("crop.html", result=result)

@app.route('/fertilizer', methods=['GET', 'POST'])
def fertilizer():
    result = None

    if request.method == 'POST':
        try:
            # Extract form data
            temp = request.form.get('temp')
            humi = request.form.get('humid')
            mois = request.form.get('mois')
            soil = request.form.get('soil')
            crop = request.form.get('crop')
            nitro = request.form.get('nitro')
            pota = request.form.get('pota')
            phosp = request.form.get('phos')
            
            input_data = [int(temp),int(humi),int(mois),int(soil),int(crop),int(nitro),int(pota),int(phosp)]




            # Make prediction
            res = ferti.classes_[model.predict([input_data])]

            result = 'Predicted Fertilizer is {}'.format(res)
        except Exception as e:
            result = "Error: {}".format(str(e))

    return render_template("fertilizer.html", result=result) 

@app.route('/cropping_techniques', methods=['GET', 'POST'])
def croppingtechniques():
    response = None 
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        crop_name = str(request.form['cropname'])

        # ph = float(request.form['ph'])

        df = pd.read_csv('C:\\Users\\bayya\\OneDrive\\Desktop\\saikiranbls\\fe.csv')

        nr = df[df['Crop'] == crop_name]['N'].iloc[0]
        pr = df[df['Crop'] == crop_name]['P'].iloc[0]
        kr = df[df['Crop'] == crop_name]['K'].iloc[0]

        n = nr - N
        p = pr - P
        k = kr - K
        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_value = temp[max(temp.keys())]
        if max_value == "N":
            if n < 0:
                key = 'NHigh'
            else:
                key = "Nlow"
        elif max_value == "P":
            if p < 0:
                key = 'PHigh'
            else:
                key = "Plow"
        else:
            if k < 0:
                key = 'KHigh'
            else:
                key = "Klow"

        fertilizer_dic = {
            'NHigh': """The N value of soil is high and might give rise to weeds.
                Please consider the following suggestions:
                ...
                (your long fertilizer suggestions here)
                """,

            # (other cases)

            'Klow': """The K value of your soil is low.
                Please consider the following suggestions:
                ...
                (your long fertilizer suggestions here)
                """
        }

        response = str(fertilizer_dic[key])

    return render_template('cropping_techniques.html', response=response)

 

if __name__ == '__main__':
    app.run(debug=True)
