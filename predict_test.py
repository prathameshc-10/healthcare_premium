import joblib
import pandas as pd
from streamlit import columns

# Load the saved model
model = joblib.load('healthcare_model.pkl')

# Create a function that the frontend will eventually call
def get_prediction(age, bmi, smoker_text):
    # Convert Yes/No text from ui to 1/0 for the model
    smoker = 1 if smoker_text.lower() == 'yes' else 0

    # Wrap the inout in dataframe so the model recognizes feature names
    input_data = pd.DataFrame([[age, bmi, smoker]], columns = ['age', 'bmi', 'smoker'])

    prediction = model.predict(input_data)

    return prediction[0]

# Test it manually
result = get_prediction(age = 30, bmi = 25, smoker_text = 'no')
print(f"Step 2 success: the loaded model predicted a premium of {result: .2f} Rs")