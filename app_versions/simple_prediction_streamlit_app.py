import streamlit as st
import pandas as pd
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


def load_pickles(model_pickle_path, label_encoder_pickle_path):
    model_pickle_opener = open(model_pickle_path, "rb")
    model = pickle.load(model_pickle_opener)

    label_encoder_pickle_opener = open(label_encoder_pickle_path, "rb")
    label_encoder_dict = pickle.load(label_encoder_pickle_opener)

    return model, label_encoder_dict


def pre_process_data(df, label_encoder_dict):
    df.drop("customerID", axis=1, inplace=True)
    for col in df.columns:
        if col in list(label_encoder_dict.keys()):
            column_le = label_encoder_dict[col]
            df.loc[:, col] = column_le.transform(df.loc[:, col])
        else:
            continue
    return df


def make_predictions(processed_df, model):
    prediction = model.predict(processed_df)
    return prediction


def generate_predictions(test_df):
    model_pickle_path = "./churn_model/churn_prediction_model.pkl"
    label_encoder_pickle_path = "./churn_model/churn_prediction_label_encoders.pkl"

    model, label_encoder_dict = load_pickles(model_pickle_path,
                                             label_encoder_pickle_path)

    processed_df = pre_process_data(test_df, label_encoder_dict)
    prediction = make_predictions(processed_df, model)
    return prediction


if __name__ == '__main__':
    # make the application
    st.title("Customer Churn Prediction")
    st.text("Enter customer data.")
    all_customers_training_data = pd.read_csv("./data/holdout_data.csv")
    all_customers_data = all_customers_training_data.drop(columns="Churn")
    chosen_customer = st.selectbox("Select the customer you are speaking to:", all_customers_training_data.loc[:, "customerID"])
    chosen_customer_data = all_customers_data.loc[all_customers_data.loc[:, 'customerID']==chosen_customer]
    # visualizing cutomer's data
    st.table(chosen_customer_data)

    # generate the prediction for the customer
    if st.button("Predict Churn"):
        pred = generate_predictions(chosen_customer_data)
        if bool(pred):
            st.text("Customer will churn!")
        else:
            st.text("Customer not predicted to churn")
