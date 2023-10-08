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

    1. Manure – adding manure is one of the simplest ways to amend your soil with nitrogen. Be careful as there are various types of manures with varying degrees of nitrogen.

    2. Coffee grinds – use your morning addiction to feed your gardening habit! Coffee grinds are considered a green compost material which is rich in nitrogen. Once the grounds break down, your soil will be fed with delicious, delicious nitrogen. An added benefit to including coffee grounds to your soil is while it will compost, it will also help provide increased drainage to your soil.

    3. Plant nitrogen fixing plants – planting vegetables that are in Fabaceae family like peas, beans and soybeans have the ability to increase nitrogen in your soil

    4. Plant ‘green manure’ crops like cabbage, corn and brocolli

    5. Use mulch (wet grass) while growing crops - Mulch can also include sawdust and scrap soft woods""",

    'Nlow': """The N value of your soil is low.
    Please consider the following suggestions:
    
    1. Add sawdust or fine woodchips to your soil – the carbon in the sawdust/woodchips love nitrogen and will help absorb and soak up and excess nitrogen.

    2. Plant heavy nitrogen feeding plants – tomatoes, corn, broccoli, cabbage and spinach are examples of plants that thrive off nitrogen and will suck the nitrogen dry.

    3. Water – soaking your soil with water will help leach the nitrogen deeper into your soil, effectively leaving less for your plants to use.

    4. Sugar – In limited studies, it was shown that adding sugar to your soil can help potentially reduce the amount of nitrogen is your soil. Sugar is partially composed of carbon, an element which attracts and soaks up the nitrogen in the soil. This is similar concept to adding sawdust/woodchips which are high in carbon content.

    5. Add composted manure to the soil.

    6. Plant Nitrogen fixing plants like peas or beans.

    7. Use NPK fertilizers with high N value.

    8. Do nothing – It may seem counter-intuitive, but if you already have plants that are producing lots of foliage, it may be best to let them continue to absorb all the nitrogen to amend the soil for your next crops.""",

    'PHigh': """The P value of your soil is high.
    Please consider the following suggestions:

    1. Avoid adding manure – manure contains many key nutrients for your soil but typically including high levels of phosphorous. Limiting the addition of manure will help reduce phosphorus being added.

    2. Use only phosphorus-free fertilizer – if you can limit the amount of phosphorous added to your soil, you can let the plants use the existing phosphorus while still providing other key nutrients such as Nitrogen and Potassium. Find a fertilizer with numbers such as 10-0-10, where the zero represents no phosphorous.

    3. Water your soil – soaking your soil liberally will aid in driving phosphorous out of the soil. This is recommended as a last ditch effort.

    4. Plant nitrogen fixing vegetables to increase nitrogen without increasing phosphorous (like beans and peas).

    5. Use crop rotations to decrease high phosphorous levels""",

    'Plow': """The P value of your soil is low.
    Please consider the following suggestions:

    1. Bone meal – a fast acting source that is made from ground animal bones which is rich in phosphorous.

    2. Rock phosphate – a slower acting source where the soil needs to convert the rock phosphate into phosphorous that the plants can use.

    3. Phosphorus Fertilizers – applying a fertilizer with a high phosphorous content in the NPK ratio (example: 10-20-10, 20 being phosphorous percentage).

    4. Organic compost – adding quality organic compost to your soil will help increase phosphorous content.

    5. Manure – as with compost, manure can be an excellent source of phosphorous for your plants.

    6. Clay soil – introducing clay particles into your soil can help retain & fix phosphorus deficiencies.

    7. Ensure proper soil pH – having a pH in the 6.0 to 7.0 range has been scientifically proven to have the optimal phosphorus uptake in plants.

    8. If soil pH is low, add lime or potassium carbonate to the soil as fertilizers. Pure calcium carbonate is very effective in increasing the pH value of the soil.

    9. If pH is high, addition of appreciable amount of organic matter will help acidify the soil. Application of acidifying fertilizers, such as ammonium sulfate, can help lower soil pH""",

    'KHigh': """The K value of your soil is high.
    Please consider the following suggestions:

    1. Loosen the soil deeply with a shovel, and water thoroughly to dissolve water-soluble potassium. Allow the soil to fully dry, and repeat digging and watering the soil two or three more times.

    2. Sift through the soil, and remove as many rocks as possible, using a soil sifter. Minerals occurring in rocks such as mica and feldspar slowly release potassium into the soil slowly through weathering.

    3. Stop applying potassium-rich commercial fertilizer. Apply only commercial fertilizer that has a '0' in the final number field. Commercial fertilizers use a three number system for measuring levels of nitrogen, phosphorous and potassium. The last number stands for potassium. Another option is to stop using commercial fertilizers all together and to begin using only organic matter to enrich the soil.

    4. Mix crushed eggshells, crushed seashells, wood ash or soft rock phosphate to the soil to add calcium. Mix in up to 10 percent of organic compost to help amend and balance the soil.

    5. Use NPK fertilizers with low K levels and organic fertilizers since they have low NPK values.

    6. Grow a cover crop of legumes that will fix nitrogen in the soil. This practice will meet the soil’s needs for nitrogen without increasing phosphorus or potassium.
    """,

    'Klow': """The K value of your soil is low.
    Please consider the following suggestions:

    1. Mix in muricate of potash or sulphate of potash
    2. Try kelp meal or seaweed
    3. Try Sul-Po-Mag
    4. Bury banana peels an inch below the soils surface
    5. Use Potash fertilizers since they contain high values potassium
    """
}

        response = str(fertilizer_dic[key])

    return render_template('cropping_techniques.html', response=response)

 

if __name__ == '__main__':
    app.run(debug=True)
