"""
Train our RNN on bottlecap or prediction files generated from our CNN.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
import numpy as np
import cv2

import time

def train(data_type, seq_length, model, saved_model=None,
          concat=False, class_limit=None, image_shape=None,
          load_to_memory=False):
    # Set variables.
    nb_epoch = 1#1000
    batch_size = 32

    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath='./data/checkpoints/' + model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5',
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir='./data/logs')

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=10)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger('./data/logs/' + model + '-' + 'training-' + \
        str(timestamp) + '.log')

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    samples_per_epoch = ((len(data.data) * 0.7) // batch_size) * batch_size

    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory(batch_size, 'train', data_type, concat)
        X_test, y_test = data.get_all_sequences_in_memory(batch_size, 'test', data_type, concat)
    else:
        # Get generators.
        generator = data.frame_generator(batch_size, 'train', data_type, concat)
        val_generator = data.frame_generator(batch_size, 'test', data_type, concat)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    # Fit!
    if load_to_memory:
        # Use standard fit.
        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[checkpointer, tb, early_stopper, csv_logger],
            #nb_epoch=nb_epoch,
            epochs=nb_epoch)
           # samples_per_epoch=samples_per_epoch)
    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            samples_per_epoch=samples_per_epoch,
            nb_epoch=nb_epoch,
            verbose=1,
            callbacks=[checkpointer, tb, early_stopper, csv_logger],
            validation_data=val_generator,
            nb_val_samples=256)

    #todo su image uzerinde goster y ytest nedir labellari neler nerden cekcen bak bakalim
    for x in range(0, 290):
        print(y_test[x])
    print(y_test[0])
    print("len y = ", y_test.shape)
    predictions = rm.model.predict(X[0:2])
    rounded = [round(x[0]) for x in predictions]
    print(rounded)
    print(np.argmax(predictions, axis=1))
    #imagebased
    #todo show image
    print("len X =", X.shape)

    best_guess = X[np.argmax(predictions)]
    print("best guess %s", best_guess)
    cv2.imshow('image', X[0,0,0])

def main():
    """These are the main training settings. Set each before running
    this file."""
    model = 'lstm'  # see `models.py` for more
    saved_model = None  # None or weights file
    class_limit = 8#None  # int, can be 1-101 or None
    seq_length = 40
    load_to_memory = True  # pre-load the sequences into memory

    # Chose images or features and image shape based on network.
    if model == 'conv_3d' or model == 'crnn':
        data_type = 'images'
        image_shape = (80, 80, 3)
    else:
        data_type = 'features'
        image_shape = None

    # MLP requires flattened features.
    if model == 'mlp':
        concat = True
    else:
        concat = False

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, concat=concat, image_shape=image_shape,
          load_to_memory=load_to_memory)

if __name__ == '__main__':
    main()
