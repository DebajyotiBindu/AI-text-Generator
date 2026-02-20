from modelling import model_training
from preprocessing import dataset
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model=model_training()
checkpoint = ModelCheckpoint(
    "models/sherlock_lstm_final.h5", 
    monitor='loss', 
    verbose=1, 
    save_best_only=True, 
    mode='min'
)
early_stopping=EarlyStopping(monitor='loss', patience=5, verbose=1, mode='min')
EPOCHS=50
history=model.fit(dataset,epochs=EPOCHS,callbacks=[checkpoint,early_stopping])
model.save("models/sherlock_lstm_final.h5")