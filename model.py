def mlp(n_in_out, n_hidden, drop_out):
    mlp = Sequential()
    mlp.add(Dense(n_hidden,
                    batch_input_shape=(None, window, n_in_out),
                    return_sequences=False, activation='relu',
                    kernel_initializer='he_normal'
                    #,kernel_regularizer=regularizers.l1(0.01)
                    ))
    mlp.add(Dropout(drop_out)) # 0.15
    mlp.add(Flatten())
    mlp.add(Dense(n_in_out))
    mlp.add(Activation("linear"))
    optimizer = Adam(lr=0.001, beta_1=0.9,beta_2=0.999,amsgrad=False)
    model.compile(loss="mean_squared_error", optimizer=optimizer,metrics=["mae"])#,rmse])# optimizer="rmsprop"
    return mlp


def rnn(n_in_out, n_hidden, drop_out):
    rnn = Sequential()
    rnn.add(LSTM(n_hidden,
                    batch_input_shape=(None, window, n_in_out),
                    return_sequences=False, activation='relu',
                    kernel_initializer='he_normal'
                    #,kernel_regularizer=regularizers.l1(0.01)
                    ))
    rnn.add(Dropout(drop_out)) # 0.15
    rnn.add(Dense(n_in_out))
    rnn.add(Activation("linear"))
    optimizer = Adam(lr=0.001, beta_1=0.9,beta_2=0.999,amsgrad=False)
    rnn.compile(loss="mean_squared_error", optimizer=optimizer,metrics=["mae"])#,rmse])# optimizer="rmsprop"
    return rnn

def lstm(n_in_out, n_hidden, drop_out):
    lstm = Sequential()
    lstm.add(LSTM(n_hidden,
                    batch_input_shape=(None, window, n_in_out),
                    return_sequences=False, activation='relu',
                    kernel_initializer='he_normal'
                    #,kernel_regularizer=regularizers.l1(0.01)
                    ))
    lstm.add(Dropout(drop_out)) # 0.15
    lstm.add(Dense(n_in_out))
    gru.add(Activation("linear"))
    optimizer = Adam(lr=0.001, beta_1=0.9,beta_2=0.999,amsgrad=False)
    lstm.compile(loss="mean_squared_error", optimizer=optimizer,metrics=["mae"])#,rmse])# optimizer="rmsprop"
    return lstm

def gru(n_in_out, n_hidden, drop_out):
    gru = Sequential()
    gru.add(GRU(n_hidden,
                    batch_input_shape=(None, window, n_in_out),
                    return_sequences=False, activation='relu',
                    kernel_initializer='he_normal'
                    #,kernel_regularizer=regularizers.l1(0.01)
                    ))
    gru.add(Dropout(drop_out)) # 0.15
    gru.add(Dense(n_in_out))
    gru.add(Activation("linear"))
    optimizer = Adam(lr=0.001, beta_1=0.9,beta_2=0.999,amsgrad=False)
    gru.compile(loss="mean_squared_error", optimizer=optimizer,metrics=["mae"])#,rmse])# optimizer="rmsprop"
    return gru
