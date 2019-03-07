import numpy as np

ori = [10000,12600,16000,22500,33000,40800,10000,10000,13800,16800,20000,24000,25500,28000,28000,25000,21500,18100,19000,21200,23400,25800,29500,30600,34000,38000,42100,39600,10000,19100,25500,31000,31000,33100,36000,40000,43000,47100,50100,47800,40000,38100,40100,42300,41100,35200,34300,33300,34800,38000,39600,40000,13000,22000,26200,30300,35200]

data = []
result = []
month_price = []

month = 6

for pointer in range(len(ori)):
    for i in range(month):
        p = i + pointer
        if p == len(ori) - 1:
            break
        month_price.append(float(ori[p]))
        if i == month - 1:
            result.append(float(ori[p + 1]))
            data.append(np.array(month_price))
            month_price = []
    
data = np.array(data) / 1000
result = np.array(result) / 1000

print(data)
print(result)

def get_value(data, result, part, index):
    value = np.concatenate(                                    
        [data[:index * part], data[(index + 1) * part:]],
        axis=0)
    target = np.concatenate(
        [result[:index * part], result[(index + 1) * part:]],
        axis=0)
    return value, target

from keras.layers.core import Dense, Dropout, Activation, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, Input
from keras.models import Model
from keras import metrics

def build_model():
    sequence_input = Input(shape=(month,))
    x = Dense(64, activation='relu')(sequence_input)
    x = Dense(64, activation='relu')(x)
    x = Dense(1)(x)

    model = Model(inputs=sequence_input, outputs=x)

    model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['mae'])
    return model
 
 
k = 5
num = len(data)
part = num // k
epochs = 200
all_scores = []
all_predict = []

predict_data = np.array([ori[-month:]]) / 1000
print(predict_data)

for i in range(k):
    val_data = data[i * part: (i + 1) * part]   
    val_target = result[i * part: (i + 1) * part]

    value, target = get_value(data, result, part,i)
    model = build_model()
    model.fit(value, target, 
              epochs=epochs, 
              batch_size=1,
             verbose=1)
    val_mse, val_mae = model.evaluate(val_data, val_target, verbose=1)   
    all_scores.append(val_mae)
    predict = model.predict(predict_data)
    all_predict.append(predict)
    print('k-fold: {} / {}, mse={}, mae={}, predict={}'.format(i + 1, k, val_mse, val_mae, predict))
    
print(all_scores)
mae = np.mean(all_scores) * 1000
print(mae)
predict_mean = np.mean(all_predict) * 1000
print(all_predict)
print('predict price:{} in max={}, min{}, mae={}'.format(predict_mean, predict_mean + mae / 2,  predict_mean - mae / 2, mae))
