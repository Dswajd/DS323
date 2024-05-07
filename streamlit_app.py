import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sklearn
print(sklearn.__version__)

# Function to load a pre-trained model from a file
def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Failed to load the model from {model_path}. Error: {e}")
        return None

# Load all models
models = {
    "K-Nearest Neighbor": load_model('knn_model.pkl'),
    "Random Forest": load_model('rf_model.pkl'),
    "Gradient Boosting": load_model('gradient_boosting_model.pkl'),
    "MLP": load_model('mlp_model.pkl')
}

# Ensure all models are loaded
for name, model in models.items():
    if model is None:
        st.error(f"Error: {name} model is not loaded correctly.")
        st.stop()

# Function to make predictions using a selected model
def make_prediction(model, input_data):
    try:
        prediction = model.predict(input_data)
        return prediction
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return []

# Setting up the Streamlit app
def main():
    st.title('Salary Prediction Model')
    st.write('This app uses machine learning models to predict the total pay for 2021 based on previous years.')

    # User inputs for the model using sliders
    tp17 = st.slider('Total Pay 2017', min_value=0, max_value=200000, step=1000, format="%d")
    tp16 = st.slider('Total Pay 2016', min_value=0, max_value=200000, step=1000, format="%d")
    tp18 = st.slider('Total Pay 2018', min_value=0, max_value=200000, step=1000, format="%d")
    tp19 = st.slider('Total Pay 2019', min_value=0, max_value=200000, step=1000, format="%d")
    tp20 = st.slider('Total Pay 2020', min_value=0, max_value=200000, step=1000, format="%d")

    # Allow user to select a model
    model_choice = st.selectbox('Choose a model:', list(models.keys()))

    # Button to make predictions
    if st.button('Predict Total Pay 2021'):
        input_df = pd.DataFrame([[tp17, tp16, tp18, tp19, tp20]],
                                columns=['Total_Pay_17', 'Total_Pay_16', 'Total_Pay_18', 'Total_Pay_19', 'Total_Pay_20'])

        # Get the prediction using the selected model
        selected_model = models[model_choice]
        prediction = make_prediction(selected_model, input_df)

        # Ensure prediction is not empty and display it
        if prediction.size > 0:
            predicted_value = float(prediction.ravel()[0])  # Convert the first element to float
            st.success(f'Predicted Total Pay for 2021: ${predicted_value:,.2f}')
        else:
            st.error('Prediction failed. Please check the input values.')

if __name__ == "__main__":
    main()
