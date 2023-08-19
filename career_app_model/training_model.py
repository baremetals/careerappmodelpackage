import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pickle


# Function to prepare data for training
def prepare_data_corrected(user_data_df, structured_data_df):
    X = []
    y = []

    for _, user_row in user_data_df.iterrows():
        features = []
        for q_idx in range(1, 21):
            response_idx = int(user_row[f'responseToQuestion{q_idx}'].split('ResponseOption')[-1])
            response_weights = structured_data_df.iloc[q_idx - 1, 8 + 3 * (response_idx - 1)::9].values
            features.extend(response_weights)

        X.append(features)

        # Calculate the average suitability score for the user's selected industries
        scores = []
        for i_idx in range(1, 6):
            industry = user_row[f'selectedIndustry{i_idx}']
            if pd.notna(industry):
                start_idx = structured_data_df.columns.get_loc(f"{industry}_Option1")
                end_idx = start_idx + 3
                weights = structured_data_df.iloc[q_idx - 1, start_idx:end_idx]
                score = np.dot(weights,
                               [1 if user_row[f'responseToQuestion{q_idx}'] == f'ResponseOption{j + 1}' else 0 for j in
                                range(3)])
                scores.append(score)

        y.append(np.mean(scores) if scores else 0)

    return np.array(X), np.array(y)


# Re-loading the data
generated_test_user_data = pd.read_csv("/mnt/data/generated_test_user_data.csv")
generated_user_data = pd.read_csv("/mnt/data/generated_user_data.csv")
model_data = pd.read_csv("/mnt/data/model_data.csv")

# Re-combining the user data
combined_user_data = pd.concat([generated_user_data, generated_test_user_data], ignore_index=True)

# Prepare the data and train the model again
X, y = prepare_data_corrected(combined_user_data, model_data)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate the model
y_pred = model.predict(X_val)

# Evaluate the model's performance
# Mean Absolute Error (MAE):
mae = mean_absolute_error(y_val, y_pred)
# Root Mean Squared Error (RMSE):
rmse = mean_squared_error(y_val, y_pred, squared=False)

# Saving the model to a pickle file
model_filename = "/mnt/data/suitability_model.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(mae, rmse)
