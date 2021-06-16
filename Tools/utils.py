import tensorflow as tf
import os
import pandas as pd
import seaborn as sns
import tensorflow as tf
from typing import *
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report






def existsfolder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 25:
        lr *= 0.5e-3
    elif epoch > 20:
        lr *= 1e-3
    elif epoch > 15:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def visualize(history: Dict, save_plot: bool = True, Path=None):
    df = pd.DataFrame(history)
    sns.set_style('whitegrid')
    plt.figure(figsize=(15, 10))
    plt.title('Model')
    plt.xlabel('Epochs #')
    plt.ylabel('Accuracy/Loss')
    sns.lineplot(data=df, markers=True, dashes=False)
    if save_plot is not False:
        plt.savefig(f"{Path}/model_performance.png")
    else:
        plt.show()



def SaveReport(y_true, y_pred, labels: List = None,Path=None):
    print(f"[REPORT]...Precisió  del model entranat: {accuracy_score(y_true, y_pred)}")
    if labels is None:
        print(classification_report(y_true, y_pred))
        with open(f"{Path}/report.txt", "w") as file:
            print(classification_report(y_true, y_pred,target_names=labels), file=file)
    else:
        print(classification_report(y_true, y_pred, target_names=labels))
        with open(f"{Path}/report.txt", "w") as file:
            print(classification_report(y_true, y_pred,target_names=labels), file=file)


def save_model_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    f = open('chars.tflite', "wb")
    f.write(tflite_model)
    f.close()





