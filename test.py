import pickle
import numpy as np

# Load the trained classifier
with open("Fertclassifier_random2.pkl", "rb") as f:
    classifier = pickle.load(f)


soil_type_mapping = {"Loamy": 1, "Sandy": 2, "Clayey": 3, "Black": 4, "Red": 5}
crop_type_mapping = {"Sugarcane": 1, "Cotton": 2, "Millets": 3, "Paddy": 4, "Pulses": 5, "Wheat": 6, "Tobacco": 7, "Barley": 8, "Oil seeds": 9, "Ground Nuts": 10, "Maize": 11}



# Function to make recommendation using the loaded classifier
def recommendation(Temparature, Humidity, Moisture, Nitrogen, Potassium, Phosphorous, Soil_Num, Crop_Num):
    features = np.array([[Temparature, Humidity, Moisture, Nitrogen, Potassium, Phosphorous, Soil_Num, Crop_Num]])
    prediction = classifier.predict(features).reshape(1,-1)
    
    return prediction[0]

# Example usage:
Temparature = 2
Humidity = 59
Moisture = 3
Nitrogen = 12
Potassium = 0
Phosphorous = 3
Soil_Type = "Sandy"
Crop_Type= "Maize"

Soil_Num = soil_type_mapping[Soil_Type]
Crop_Num = crop_type_mapping[Crop_Type]
predict = recommendation(Temparature, Humidity, Moisture, Nitrogen, Potassium, Phosphorous, Soil_Num, Crop_Num)
print(predict)
