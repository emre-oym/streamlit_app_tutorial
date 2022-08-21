# Streamlit App Tutorial - Machine Learning Front-End for Customer Churn Predictor
This tutorial walks through how to make your first [streamlit](https://docs.streamlit.io/) front-end application for interaction with a deployed machine learning model, and how to deploy it using streamlit cloud.

## About the Project 
The notebook [streamlit_app_instructions.ipynb](streamlit_app_instructions.ipynb) walks through the steps for creating your first streamlit application, with code snippets and images.

In the end, the goal is to make an interactive front-end dashboard for interacting with the machine learning model. The final application will look like this:

<img src="./images/app_screenshot_5.png" width="200"/>.

#### You can interact with the final application [here](https://rachelkberryman-churn-predictor-prediction-streamlit-app-coqysw.streamlitapp.com/).

Each step is also shown in a streamlit app file that can be run on its own: these are available in the folder [./app_versions/](./app_versions/).

The data used is the [Kaggle Telco Customer Churn dataset](https://www.kaggle.com/code/mechatronixs/telco-churn-prediction-feature-engineering-eda/data). The instructions first walk through making a simple streamlit app to show some analysis of the dataset, such as the distribution of customers who churned. 

Next, a machine learning model is trained to predict the churn. The model training code can be found in the [notebooks folder](./notebooks/model_training.py). 

Finally, the streamlit app is updated incrementally: first to make a prediction on one row of the holdout test dataset, then, to make predictions based on user input.

## Getting started
To get started, clone this repository, and go into the repository:

` cd streamlit_app_tutorial`

Next, start an instance of `jupyer lab` or `jupyter notebook` to run the notebook `streamlit_app_instructions.ipynb`.

This notebook contains all the instructions needed to start your streamlit app: including working with a virtual environment via `pipenv`.

Happy coding!