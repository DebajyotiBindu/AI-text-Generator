from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input
from preprocessing import SEQ_LENGTH, tokenizer

def model_training():
    vocab_size=len(tokenizer.get_vocabulary())
    model=Sequential()
    model.add(Input(shape=(SEQ_LENGTH,)))
    model.add(Embedding(input_dim=vocab_size,output_dim=128,input_length=SEQ_LENGTH))
    model.add(LSTM(128,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size,activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
    return model

if __name__=="__main__":
    model=model_training()
    model.summary()