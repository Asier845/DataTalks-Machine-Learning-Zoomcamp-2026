import pickle
import pandas as pd

# Load the trained model
with open("pipeline.bin", "rb") as f:
    model = pickle.load(f)

# Define the lead
lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41
}

probabilities = model.predict_proba([lead])


# Get the probability for the positive (conversion) class
classes = list(model.classes_)
positive_index = classes.index(1) if 1 in classes else len(classes) - 1
probability = probabilities[0][positive_index]

print(f"Conversion probability: {probability:.3f}")