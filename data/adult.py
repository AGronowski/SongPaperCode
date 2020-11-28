import numpy as np
import pandas as pd
import tensorflow as tf
import pickle as pkl

print(tf.__version__)
tfd = tf.contrib.distributions


def save_adult_datasets():
    adult_data = pd.read_csv('adult.data.txt', header=None, sep=', ').as_matrix()
    adult_test = pd.read_csv('adult.test.txt', header=None, sep=', ').as_matrix()

    #remove each row with missing data represented by question mark
    def remove_question(df):
        idx = np.ones([df.shape[0]], dtype=np.int32)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                try:
                    if '?' in df[i, j]:
                        idx[i] = 0
                except TypeError:
                    pass
        df = df[np.nonzero(idx)]
        return df

    #removes the K in 50K in the rightmost column
    #not sure why this is needed
    def remove_dot(df):
        for i in range(df.shape[0]):
            df[i, -1] = df[i, -1][:-1]
        return df

    #create list containing all labels for every category
    #if column contains strings than all unique strings are listed,
    #else the median is listed
    def gather_labels(df):
        labels = []
        for j in range(df.shape[1]):
            if type(df[0, j]) is str:
                labels.append(np.unique(df[:, j]).tolist())
            else:
                labels.append(np.median(df[:, j]))
        return labels

    adult_data = remove_question(adult_data)
    adult_test = remove_dot(remove_question(adult_test))
    adult_labels = gather_labels(adult_data)

    def transform_to_binary(df, labels):
        d = np.zeros([df.shape[0], 102])
        u = np.zeros([df.shape[0], 1])
        y = np.zeros([df.shape[0], 1])

        #u is gender, y is income >50 K, d is all others as 1-hot encodings
        idx = 0
        for j in range(len(labels)):
            #list of all unique strings in a column rather than median
            if type(labels[j]) is list:
                #non binary feature
                #1 hot encoding
                if len(labels[j]) > 2:
                    for i in range(df.shape[0]):
                        d[i, idx + int(labels[j].index(df[i, j]))] = 1
                    idx += len(labels[j])

                #gender is protected attribute
                #finds gender feature
                elif 'ale' in labels[j][0]:
                    for i in range(df.shape[0]):
                        u[i] = int(labels[j].index(df[i, j]))

                #binary features excluding gender
                #only option is greater or less than 50K
                else:
                    for i in range(df.shape[0]):
                        y[i] = int(labels[j].index(df[i, j]))
            else:
                #1 if datapoint is above median, 0 otherwise
                for i in range(df.shape[0]):
                    d[i, idx] = float(df[i, j] > labels[j])
                idx += 1
        return d.astype(np.bool), u.astype(np.bool), y.astype(np.bool)  # observation, protected, label

    adult_binary = transform_to_binary(adult_data, adult_labels)
    adult_test_binary = transform_to_binary(adult_test, adult_labels)

    with open('adult_binary.pkl', 'wb') as f:
        pkl.dump(adult_binary, f)
    with open('adult_test_binary.pkl', 'wb') as f:
        pkl.dump(adult_test_binary, f)


def create_adult_datasets(batch=64):
    import os
    cwd = os.getcwd()  # Get the current working directory (cwd)
    print('asdadsadsdsadsda')
    print(cwd)


    with open('adult_binary.pkl', 'rb') as f:
        ab = pkl.load(f)
    with open('adult_test_binary.pkl', 'rb') as f:
        atb = pkl.load(f)

    #contains 3 arrays corresponding to d, u , y
    #30162 data points
    adult_binary = tuple([a.astype(np.float32) for a in ab])
    adult_test_binary = tuple([a.astype(np.float32) for a in atb])

    #prefetch 64 batches of 64 elements
    #https: // www.tensorflow.org / api_docs / python / tf / data / Dataset  # prefetch
    #shuffle entire dataset then split into batches
    train = tf.data.Dataset.from_tensor_slices(adult_binary).shuffle(adult_binary[0].shape[0]).batch(batch).prefetch(batch)
    test = tf.data.Dataset.from_tensor_slices(adult_test_binary).batch(batch).prefetch(batch)

    #p(u) is the distibution of the binary sensitive gender attribute
    #probs is probability of the positive event of male
    #p = 0.67568463
    pu = tfd.Bernoulli(probs=np.mean(adult_binary[1]))
    return train, test, pu


if __name__ == '__main__':
    print('saving')
    save_adult_datasets()
