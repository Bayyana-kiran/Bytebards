 # Hackathon at GRIET(ACI challenge) - AICTE
 
 
 # ECOSOIL - AI BASED FARMING

This repository contains the code for a crop recommendation system that uses machine learning models to predict the best crop to cultivate based on soil conditions and environmental factors. The system also provides fertilizer recommendations and cropping techniques to optimize crop yield.

## Prerequisites

To run the code, you will need the following:

* Python 3.6 or later
* NumPy
* Pandas
* Scikit-learn
* Flask
* Pickle

## Installation

To install the required dependencies, run the following command in your terminal:

```
pip install -r requirements.txt
```

## Usage

To run the crop recommendation system, follow these steps:

1. Clone the repository to your local machine.
2. Open the `app.py` file in a text editor.
3. Replace the `model.pkl`, `standscaler.pkl`, `minmaxscaler.pkl`, and `classifier.pkl` file paths with the paths to the corresponding files on your local machine.
4. Save the `app.py` file.
5. Open a terminal window and navigate to the directory where the `app.py` file is located.
6. Run the following command to start the Flask application:

```
flask run
```

The crop recommendation system will now be running on `localhost:5000`. You can access the system by opening a web browser and navigating to `http://localhost:5000`.

## Features

The crop recommendation system has the following features:

* **Crop recommendation:** The system uses a machine learning model to predict the best crop to cultivate based on soil conditions and environmental factors.
* **Fertilizer recommendation:** The system provides fertilizer recommendations based on the predicted crop and soil conditions.
* **Cropping techniques:** The system provides cropping techniques to optimize crop yield.

## Screenshots


## Conclusion

The crop recommendation system is a powerful tool that can help farmers optimize their crop yields. The system is easy to use and can be accessed from any device with an internet connection.

