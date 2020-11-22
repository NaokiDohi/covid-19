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


