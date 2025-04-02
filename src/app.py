import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas as pd
import joblib
import json

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
                import get_additional_features as gef

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

                # Create price trends chart
                st.subheader("Price Trends")
                # Generate a range of dates for the next 30 days
                future_dates = pd.date_range(start=date, periods=360)
                price_data = []

                for d in future_dates:
                    # Extract additional features for each date
                    quarter = (d.month - 1) // 3 + 1
                    year = d.year

                    # Create input array for prediction
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

                    for col in categorical_columns:
                        input_df[col] = input_df[col].astype('category')

                    predicted_price = model.predict(input_df)[0]

                    # Append the date and predicted price to the price data
                    price_data.append((d, predicted_price))

                # Create a DataFrame for the price data
                price_df = pd.DataFrame(price_data, columns=["Date", "Price"])

                # Convert Date column to numerical values for smoothing
                price_df["Date_ordinal"] = price_df["Date"].map(lambda x: x.toordinal())

                # Fit a smooth curve using interpolation
                x = np.linspace(
                    price_df["Date_ordinal"].min(), price_df["Date_ordinal"].max(), 300
                )
                prices = np.interp(x, price_df["Date_ordinal"], price_df["Price"])

                # Convert ordinal back to datetime for better labeling
                dates = [datetime.date.fromordinal(int(i)) for i in x]

                # Plot
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(x=dates, y=prices, ax=ax, linewidth=2, color="dodgerblue")

                ax.set_title("Flight Price Trends", fontsize=14)
                ax.set_xlabel("Date", fontsize=12)
                ax.set_ylabel("Price ($)", fontsize=12)
                plt.xticks(rotation=45)

                # Display in Streamlit
                st.pyplot(fig)


if __name__ == "__main__":
    main()
