from tensorflow.keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
        rec = tp/(K.sum(K.round(K.clip(y_true, 0, 1))) + K.epsilon())
        return rec

    def precision(y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
        prec = tp/(K.sum(K.round(K.clip(y_pred, 0, 1))) + K.epsilon())
        return prec
    
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec + rec + K.epsilon()))

def acc(y_true, y_pred):
    return (y_true.argmax(axis=1)==y_pred.argmax(axis=1)).sum()/len(y_true)