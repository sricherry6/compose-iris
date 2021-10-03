import os
import pickle
from sklearn.naive_bayes import GaussianNB

# define the class encodings and reverse encodings
classes = {0: "Non-Diabetic", 1: "Diabetic"}
r_classes = {y: x for x, y in classes.items()}

# function to process data and return it in correct format
def process_data(data):
    processed = [
        {
            "number_of_times_pregnent": d.number_of_times_pregnent,
            "plasma_glucose": d.plasma_glucose,
            "blood_pressure": d.blood_pressure,
            "skinfold_thickness": d.skinfold_thickness,
            "serum_insulin": d.serum_insulin,
            "bmi": d.bmi,
            "diabetes_pf": d.diabetes_pf,
            "age": d.age,
            "isdiabetic": d.isdiabetic
        }
        for d in data
    ]

    return processed
