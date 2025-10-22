from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf  

MAX_LEN=128
MODEL_PATH = "models/bert_finetuned"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)

def predict_text(text):
    encoding = tokenizer(
        text, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="tf"
    )
    inputs = dict(encoding)
    outputs = model.predict(inputs)
    logits = outputs.logits
    pred = tf.nn.softmax(logits,axis=1)
    label = int(tf.argmax(pred, axis=1))
    prob = float(tf.reduce_max(pred))
    return label, prob