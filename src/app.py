import streamlit as st
import pandas as pd
import joblib
import json
import get_additional_features as gef

# Streamlit UI
def main():

    # Set page config
    st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️", layout="centered", initial_sidebar_state="collapsed")

    # Load the trained model
    model = joblib.load("models/lgb_model.pkl")

    # load the encoded df
    with open("data/processed/df_encoded.json", "r") as enc_file:
        df_encodings = json.load(enc_file)
        
    # Load Styles
    with open("styles/style.css", "r") as css_file:
        css = css_file.read()
        
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    
    st.title("Flight Price Predictor")

    # Set routes
    routes = [entry["route"] for entry in df_encodings]

    # Extract unique departure airports from the routes
    departure_airports = sorted(list(set(route.split("-")[0] for route in routes)))

    date = st.date_input("Select a date")


    valid_arrivals = []

    # Departure Airport Selection
    airport_1 = st.selectbox("Departure Airport", departure_airports, placeholder="Select a departure airport", index=None)

    if airport_1 and airport_1 != "Select a departure airport":
        valid_arrivals = sorted(set([route.split("-")[1] for route in routes if route.startswith(airport_1 + "-")]))

        # Arrival Airport Selection
        airport_2 = st.selectbox("Arrival Airport", valid_arrivals, placeholder="Select an arrival airport", index=None)

        if airport_2 and airport_2 != "Select an arrival airport":
            passengers = st.number_input("Number of Passengers", min_value=1, max_value=10, value=1, step=1, help="Select the number of passengers")

            # Predict Price Button
            if st.button("Predict Price"):
                # Extract additional features from the date
                quarter = (date.month - 1) // 3 + 1
                year = date.year
                route = airport_1 + "-" + airport_2
                distance = gef.get_additional_features("distance", route, df_encodings)
                carrier_lg = gef.get_additional_features("carrier_lg", route, df_encodings)
                carrier_low = gef.get_additional_features("carrier_low", route, df_encodings)

                # Create an input array based on the model features
                input_df = pd.DataFrame({
                    "year": [year],
                    "quarter": [quarter],
                    "airport_1": [airport_1],
                    "airport_2": [airport_2],
                    "distance": [distance],
                    "carrier_lg": [carrier_lg],
                    "carrier_low": [carrier_low],
                    "route": [route]
        })

                # Determine which columns are categorical
                categorical_columns = [
                    "airport_1",
                    "airport_2",
                    "carrier_lg",
                    "carrier_low",
                    "quarter",
                    "year",
                    "route"
                ]

                for col in categorical_columns:
                    input_df[col] = input_df[col].astype('category')

                predicted_price = model.predict(input_df)[0]

                total_price = predicted_price * passengers
                
                st.subheader("Predicted Flight Price")
                st.success(f"${total_price:.2f}")


if __name__ == "__main__":
    main()
