# nlp-ChnSentiCorp

## 1、下载数据

https://www.kaggle.com/datasets/kaggleyxz/chnsenticorp?resource=download

下载后保存至`datasets`目录下

## 2、加载数据

使用tab分割模式读取数据集

```py
def load_dataset(tabel='train'):
    file = './datasets/'+tabel+'.tsv'
    df = pd.read_csv(file, sep='\t')
    df = df.dropna()
    texts = df['text_a'].astype(str).tolist()
    if tabel == 'test':
        labels = None
    else:
        labels = df['label'].astype(int).tolist()
        
    return texts, labels
```

从网上下载停用词文件，加载停用词

```py
def load_stopwords(filep='./datasets/stopwords.txt'):
    with open(filep,encoding='utf-8') as f:
        return set([w.strip() for w in f.readlines()])
stopwords = load_stopwords()
```

使用`jieba`进行中文句子分词

```py
def chinese_tokenizer(text):
    return [w for w in jieba.lcut(text) if w not in stopwords and w.strip()]
def text_cut(texts):
    return [' '.join(chinese_tokenizer(text)) for text in texts]
```

使用`Tokenizer`将中文分词转化为词向量，并进行截断处理

```py
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(train_cut)
train_sequences = tokenizer.texts_to_sequences(train_cut)
test_sequences = tokenizer.texts_to_sequences(test_cut)
x_train = pad_sequences(train_sequences,maxlen=MAX_SEQ_LEN)
y_train = np.array(train_labels)
x_test = pad_sequences(test_sequences, maxlen=MAX_SEQ_LEN)
y_test = np.array(test_labels)
```

使用`tf.data.Dataset`将数据进行批次化并进行打乱

```py
db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
db_train = db_train.shuffle(1000).batch(BATCH_SIZE,drop_remainder=True)
```

## 3、建立模型

建立LSTM模型，使用adam优化器

```py
model = Sequential()
model.add(Embedding(MAX_VOCAB_SIZE,EMBEDDING_DIM,input_length=MAX_SEQ_LEN))
model.add(LSTM(128,dropout=0.3))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

训练6轮后在测试集上的准确率达到86.55%

```text
18/18 [==============================] - 0s 19ms/step - loss: 0.5456 - accuracy: 0.8655
```

## 4、BERT微调

安装`transformer`库  
从库中导入`BertTokenizer`，`TFBertForSequenceClassification`  
由于本地无法访问hugging face网站在线加载模型需要的文件，因此可以通过手动下载到本地
https://huggingface.co/google-bert/bert-base-chinese/tree/main

下载后保存到`models/bert-base-chinese/`，用于加载分词器和模型

```py
MODEL_NAME = "./models/bert-base-chinese/"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = TFBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
```

加载数据后使用tokenizer进行分词编码后，才能作为模型的输入

```py
def encode_texts(texts, labels):
    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="tf"
    )
    return tf.data.Dataset.from_tensor_slices((
        dict(encodings),
        labels
    ))
train_ds = encode_texts(train_texts, train_labels).shuffle(1000).batch(BATCH_SIZE)
test_ds = encode_texts(test_texts,test_labels).batch(BATCH_SIZE)
```

模型训练后在测试集上的准确率达到94.33%

```py
history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)
```

```text
Epoch 1/3
572/572 [==============================] - 411s 697ms/step - loss: 0.3200 - accuracy: 0.8717 - val_loss: 0.1991 - val_accuracy: 0.9225
Epoch 2/3
572/572 [==============================] - 347s 607ms/step - loss: 0.1402 - accuracy: 0.9485 - val_loss: 0.1961 - val_accuracy: 0.9400
Epoch 3/3
572/572 [==============================] - 124s 217ms/step - loss: 0.0635 - accuracy: 0.9791 - val_loss: 0.2026 - val_accuracy: 0.9433
```

将微调后的模型保存

```py
model.save_pretrained("models/bert_finetuned")
tokenizer.save_pretrained("models/bert_finetuned")
```

## 5、Fastapi+Docker部署

### 定义fastapi

安装兼容版本的fastapi和uvicorn库

```text
fastapi==0.75.0
uvicorn==0.22.0
```

创建一个FastAPI对象，并定义接口函数  
main.py

```py
from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_text

app = FastAPI(title="BERT Chinese Text Classification API")

# 输入数据格式
class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "BERT API running successfully"}

@app.post("/predict")
def predict(input_data: TextInput):
    text = input_data.text
    label, prob = predict_text(text)
    return {"text": text, "label": label, "probability": prob}
```

加载微调过的BERT模型，并进行预测  
predict.py

```py
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
```

使用uvicorn命令运行

```bash
uvicorn main:app --reload
```

然后在浏览器访问

API:`http://127.0.0.1:8000`  
交互式：`http://127.0.0.1:8000/docs`  

### Docker部署

安装Docker Desktop，并安装一个wsl发行版本  
创建Dockerfile文件

```docker
# 使用轻量级 Python 镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 加载依赖文件
COPY requirements.txt /app

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 避免 TensorFlow 提示警告
ENV TF_CPP_MIN_LOG_LEVEL=2

# 复制项目文件
COPY . /app

# 默认使用 UTF-8 编码
ENV PYTHONUNBUFFERED=1

# 暴露 FastAPI 默认端口
EXPOSE 8000

# 启动 FastAPI 服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

在Dockerfile文件目录下运行，创建docker镜像

```bash
docker build -t bert-fastapi .
```

运行并创建容器

```bash
docker run -d -p 8000:8000 --name bert_server bert-fastapi
```

这样就可以在内网访问`http://localhost:8000`，要是想在公网访问，可以在wsl安装cloudflare创建tunnel将本地环境暴露给公网

```bash
cloudflare tunnel --url http://localhost:8000
```

终端会输出类似`https://*.trycloudflare.com`地址，通过公网访问这个地址就可以了
