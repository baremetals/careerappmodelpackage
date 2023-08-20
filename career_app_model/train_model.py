from config.core import config
from processing.data_manager import load_dataset, save_model
from processing.prepare_data import prepare_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def run_training() -> None:
    # read training data
    model_data = load_dataset(file_name=config.app_config.training_data_file)

    # read user data
    user_data = load_dataset(file_name=config.app_config.test_data_file)

    # Prepare the data and train the model again
    X, y = prepare_data(user_data, model_data)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config.model_config.test_size, random_state=config
                                                      .model_config.random_state)

    # Train the model
    model = RandomForestRegressor(n_estimators=config.model_config.n_estimators, random_state=config.model_config
                                  .random_state)

    model.fit(X_train, y_train)

    # Validate the model
    y_prediction = model.predict(X_val)

    # Evaluate the model's performance
    mae = mean_absolute_error(y_val, y_prediction)
    rmse = mean_squared_error(y_val, y_prediction, squared=False)

    print(mae, rmse)

    # persist trained model
    save_model(model_to_persist=model)


if __name__ == "__main__":
    run_training()
