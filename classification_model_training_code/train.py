import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
from data_loader import prepare_labels, create_dataloaders
from model import load_pretrained_model

IMAGE_SIZE = (299, 299, 3)
EPOCHS = 10
BATCH_SIZE = 16
DATA_DESRC_DIR = 'data/toy-art-data/balanced_labels_ru.csv'
ART_DIR = 'data/toy-art-data/toy_dataset'
PRETRAINED_MODEL_PATH = "results_v2/model_resnet_ru_v2.keras"
MODEL_NAME = "model_resnet_ru.keras"

def plot_graphs(history, string, model_name):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend(['train_'+string, 'val_'+string])
    plt.savefig(model_name + string + '.png')
    np.savetxt("train_"+string+".csv", history.history[string], delimiter=",")
    np.savetxt("val_"+string+".csv", history.history['val_'+string], delimiter=",")
    plt.show()

def train_model(model, training_dataloader, test_dataloader, model_name='model.keras', epochs=10):
    checkpoint = ModelCheckpoint(
        model_name,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )

    earlystopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0, 
        patience=1, 
        verbose=1, 
        restore_best_weights=True
    )
    
    history = model.fit(
        training_dataloader, 
        epochs=epochs,
        validation_data=test_dataloader,
        callbacks=[checkpoint, earlystopping]
    )
    return history

def main():
    labels_df, y, binarizer = prepare_labels(DATA_DESRC_DIR)
    training_dataloader, test_dataloader = create_dataloaders(
        labels_df, y, binarizer, ART_DIR, IMAGE_SIZE, BATCH_SIZE
    )
    
    model = load_pretrained_model(
        pretrained_model_path=PRETRAINED_MODEL_PATH,
        input_shape=IMAGE_SIZE,
        output_shape=y.shape[1]
    )
    model.summary()
    
    history = train_model(
        model=model,
        training_dataloader=training_dataloader,
        test_dataloader=test_dataloader,
        model_name=MODEL_NAME,
        epochs=EPOCHS
    )
    
    plot_graphs(history, "loss", MODEL_NAME)
    plot_graphs(history, "precision", MODEL_NAME)
    plot_graphs(history, "binary_accuracy", MODEL_NAME)
    
    y_test = test_dataloader.y[:100]
    y_pred = model.predict(test_dataloader[0:100]).round()
    print(classification_report(y_test, y_pred, target_names=binarizer.classes_))

if __name__ == "__main__":
    main()
