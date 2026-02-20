import tensorflow as tf
import re
import numpy as np

FILE_PATH=r"data/data.txt"
def clean_text(text):
    with open(FILE_PATH,'r',encoding='utf-8') as f:
        text=f.read()
    text=text.lower()
    text=text.replace('\n',' ')
    text=re.sub(r'[^\w\s]','',text)
    text=" ".join(text.split())
    return text

cleaned_text=clean_text(FILE_PATH)

VOCAB_SIZE=10000
tokenizer=tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE,output_mode='int')
word_list=cleaned_text.split()
tokenizer.adapt(word_list)

tokens=tokenizer(word_list)
tokens=tf.reshape(tokens,[-1])

SEQ_LENGTH=10
BATCH_SIZE=64
def create_dataset(tokens,seq_length,batch_size):
    ds=tf.data.Dataset.from_tensor_slices(tokens)
    ds=ds.window(seq_length+1,shift=1,drop_remainder=True)
    ds=ds.flat_map(lambda w: w.batch(seq_length + 1))
    def split_fn(seq):
        return seq[:-1],seq[-1]
    ds=ds.map(split_fn)
    ds=ds.shuffle(10000).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds

dataset=create_dataset(tokens,SEQ_LENGTH,BATCH_SIZE)

for x,y in dataset.take(1):
    vocab=tokenizer.get_vocabulary()
    print("Input (Indices):", x[0].numpy())
    print("Input (Words):  ", [vocab[i] for i in x[0].numpy()])
    target_idx = y[0].numpy()
    print("Target (Index): ", target_idx)
    print("Target (Word):  ", vocab[target_idx])
