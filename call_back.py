def early_stopping():
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=1)
    return early_stopping

def lr_schedul(epoch):
    x = 0.001
    if epoch >= 300:
        x = 0.0005
    return x

def lr_decay():
    lr_decay = LearningRateScheduler(
    lr_schedul,
    # verbose=1で、更新メッセージ表示。0の場合は表示しない
    verbose=0,
    )
    return lr_decay
