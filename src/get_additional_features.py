import pandas as pd

def get_additional_features(feature, route, df_encoded):

    # Convert list of dictionaries to DataFrame
    df_encoded = pd.DataFrame(df_encoded)

    # Find a row with a matching `route_encoded` value
    matched_row = df_encoded[df_encoded["route"] == route]

    return matched_row.iloc[0][feature]