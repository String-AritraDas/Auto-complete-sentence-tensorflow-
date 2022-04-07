import random
from os.path import exists
import numpy as np
from keras import Sequential
from keras.layers import LSTM, Embedding, Bidirectional, Dense
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


def _data_setup(oov_token, _check_point_file_path, _vocab_size=10000):
    check_point_path = _check_point_file_path

    vocab_size = 10000
    data = open('__data__\\training_datasets.txt').read()
    corpus = data.lower().split("\n")

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    input_sequence = []

    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequence.append(n_gram_sequence)

    max_length = max([len(x) for x in input_sequence])
    input_sequence = np.array(pad_sequences(input_sequence, maxlen=max_length, padding='pre'))

    xs = input_sequence[:, :-1]
    labels = input_sequence[:, -1]
    ys = to_categorical(labels, num_classes=total_words)

    return total_words, xs, ys, check_point_path, max_length, tokenizer


def _create_model(total_words,
                  check_point_path,
                  max_length,
                  xs,
                  ys,
                  _embedding_output_dim=240,
                  _activation_function='softmax',
                  lstm_rnn=150,
                  loss_function="categorical_crossentropy",
                  adam_lr=0.01,
                  epochs=5,
                  training=True):

    model = Sequential()
    model.add(Embedding(total_words, _embedding_output_dim, input_length=max_length - 1))
    model.add(Bidirectional(LSTM(lstm_rnn)))
    model.add(Dense(total_words, activation=_activation_function))
    if not training:
        print("...Loading model...")
        try:
            model.load_weights(check_point_path)
        except:
            print("No model weight data found! Please train the model before testing")
    else:
        model.load_weights(check_point_path)
        adam = Adam(learning_rate=adam_lr)
        model.compile(optimizer=adam, loss=loss_function, metrics=['accuracy'])
        cp_callback = ModelCheckpoint(check_point_path, verbose=1, save_weights_only=True)
        print("...Start training...")
        model.fit(xs, ys, epochs=epochs, callbacks=cp_callback)
        print("...Model summary...")
        print(model.summary())
        print("...Model configuration...")
        print(model.get_config())

    return model


def test_code(model, tokenizer, max_length):
    while True:
        send_text = input("Enter a line: ")
        if send_text == 'exit':
            break
        #next_words_str = input("Enter number of words to generate: ")
        next_words = input("Enter word to genarate: ")
        count = 0
        for _ in range(int(next_words)):
            count += 1
            token_list_element = tokenizer.texts_to_sequences([send_text])[0]
            token_list = pad_sequences([token_list_element], maxlen=max_length - 1, padding='pre')
            predicted_model = model.predict(token_list)
            predicted = np.argmax(predicted_model, axis=1)
            output_word = ""
            if count == int(next_words):
                if output_word == 'the' or output_word == 'a' or output_word == 'of':
                    output_word == ''

            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break

            send_text += " " + output_word
        print(send_text)


if __name__ == '__main__':
    while True:
        training = True

        print("_________________________________Welcome to Tensor Bot_________________________________")
        print("_______________________________________________________________________________________")

        while True:
            condition = input("Type 'train' to train model and 'test' to test model ['x' to exit]: ")
            if condition == "train":
                training = True
                break
            elif condition == "test":
                training = False
                break
            elif condition == "x":
                exit(0)
            else:
                print("Type 'test' or 'train'")

        if training:
            print("...Training mode...")
            print("\n...Initializing data model for you...\n")
            _oov_token = input("OOV token: ")
            _vocab_size = input("Vocabulary size: ")
            file_exists = exists("_model_personal_data\\dt.cpl")
            if file_exists:
                _check_point_path = open("_model_personal_data\\dt.cpl", 'r').read()
            else:
                _check_point_path = input("Check point path [must contain file name like name.ckpt]: ")
                open("_model_personal_data\\dt.cpl", 'w+').write(_check_point_path)

            total_words, xs, ys, check_point_path, max_length, tokenizer = _data_setup(oov_token=_oov_token,
                                                                                       _check_point_file_path=_check_point_path, _vocab_size=int(_vocab_size))
            print("\n...Creating model for you...\n")
            _output_dim = input("Embedding hidden layer output dim: ")
            _act_func = input("Activation function name: ")
            _lstm_rnn = input("Lstm rnn unites: ")
            _loss_func_name = input("Loss function name: ")
            _adam_lr = input("Optimizer (default: adam) learning rate: ")
            _epochs = input("Epochs: ")
            try:
                _create_model(total_words, check_point_path, max_length, xs, ys, int(_output_dim), _act_func, int(_lstm_rnn), _loss_func_name, float(_adam_lr),
                            epochs=int(_epochs), training=True)
            except:
                print("Given values are wrong!")
        else:
            print("...Testing mode...")
            print("\n...Initializing data model for you...\n")
            _oov_token = input("OOV token: ")
            _vocab_size = input("Vocabulary size: ")
            file_exists = exists("_model_personal_data\\dt.cpl")
            if file_exists:
                _check_point_path = open("_model_personal_data\\dt.cpl", 'r').read()
            else:
                _check_point_path = input("Check point path [must contain file name like name.ckpt]: ")
                open("_model_personal_data\\dt.cpl", 'w+').write(_check_point_path)
            total_words, xs, ys, check_point_path, max_length, tokenizer = _data_setup(oov_token=_oov_token,
                                                                                       _check_point_file_path=_check_point_path,
                                                                                       _vocab_size=int(_vocab_size))
            print("\n...Creating model for you...\n")
            _output_dim = input("Embedding hidden layer output dim: ")
            _act_func = input("Activation function name: ")
            _lstm_rnn = input("Lstm rnn unites: ")
            model = _create_model(total_words, check_point_path, max_length, xs, ys, int(_output_dim), _act_func, int(_lstm_rnn), training=False)
            test_code(model=model, max_length=max_length, tokenizer=tokenizer)
