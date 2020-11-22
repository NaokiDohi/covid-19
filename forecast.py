def past_predict(study_data):
    predicted_past_data = model.predict(study_data)
    train_inverse= scaler.inverse_transform(predicted_past_data)
    return train_inverse

def test_predict(upcoming_future):
    upcoming_future = upcoming_future
    predictions = train_normalized[-window:].tolist()
    predictions = np.array(predictions).reshape(-1, window, 1)
    for i in range(upcoming_future):
        predicted_future = model.predict(predictions)
        # with open("in_out.txt",mode="a", encoding= "utf-8") as f:
            # f.write("input to model:" + str(predictions) )
            # f.write("output from model:" + str(predicted_future) )
    predictions = predictions.tolist()
    predictions = np.append(predictions,predicted_future)
    predictions = predictions[-window:]
    predictions = np.array(predictions).reshape(-1, window, 1)
    predictions_infected_pepole = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
    return predictions_infected_pepole

def long_predict(long_term_future):
    predictions2 = train_normalized[-window:].tolist()
    predictions2 = np.array(predictions2).reshape(-1, window, 1)
    long_term_predictions = []
    for i in range(long_term_future):
        predicted_long_term_future = model.predict(predictions2)
        long_term_predictions.append(predicted_long_term_future)
            # with open("in_out_long_term.txt",mode="a", encoding= "utf-8") as f:
            #     f.write("input to model:" + str(predictions) )
            #     f.write("output from model:" + str(predicted_future) )
    predictions2 = predictions2.tolist()
    predictions2 = np.append(predictions2, predicted_long_term_future)
    predictions2 = predictions2[-window:]
    predictions2 = np.array(predictions2).reshape(-1, window, 1)
    predictions_infected_pepole_long_term = scaler.inverse_transform(np.array(long_term_predictions).reshape(-1,1))
    return predictions_infected_pepole_long_term

def eval_func(train_inverse, test_original_data, predictions_infected_pepole):
    train_data = train_original_data[7:]
    train_mae = mean_absolute_error(train_data, train_inverse)
    train_mse = mean_squared_error(train_data, train_inverse)
    train_rmse = np.sqrt(mean_squared_error(train_data, train_inverse))
    train_r2 = r2_score(train_data, train_inverse)
    test_mae = mean_absolute_error(test_original_data, predictions_infected_pepole)
    test_mse = mean_squared_error(test_original_data, predictions_infected_pepole)
    test_rmse = np.sqrt(mean_squared_error(test_original_data, predictions_infected_pepole))
    test_r2 = r2_score(test_original_data, predictions_infected_pepole)
    return train_mae, train_mse, train_rmse, train_r2, test_mae, test_mse, test_rmse, test_r2
