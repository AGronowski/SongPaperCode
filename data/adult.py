import numpy as np
import pandas as pd
import tensorflow as tf
import pickle as pkl

tfd = tf.contrib.distributions


def save_adult_datasets():
    # as_matrix converts dataframe to numpy array
    # .values should be used instead
    adult_data = pd.read_csv('adult.data.txt', header=None, sep=', ').values
    adult_test = pd.read_csv('adult.test.txt', header=None, sep=', ').values

    # remove each row with missing data represented by question mark
    def remove_question(df):
        idx = np.ones([df.shape[0]], dtype=np.int32)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                try:
                    if '?' in df[i, j]:
                        idx[i] = 0
                except TypeError:
                    pass
        # np.nonzero(idx) is a list of indices of rows with no missing data
        df = df[np.nonzero(idx)]
        return df

    # removes the K in 50K in the rightmost column
    # don't think this function is actually needed
    def remove_dot(df):
        for i in range(df.shape[0]):
            df[i, -1] = df[i, -1][:-1]
        return df

    # create list containing all labels for every category
    # if column contains strings then all unique strings are listed,
    # else the median is listed
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

        # u is gender, y is income >50 K, d is all others as 1-hot encodings
        idx = 0
        for j in range(len(labels)):
            # list of all unique strings in a column rather than median
            if type(labels[j]) is list:
                # non binary feature
                # 1 hot encoding
                if len(labels[j]) > 2:
                    # .index returns the index of where in the list the argument is
                    for i in range(df.shape[0]):
                        d[i, idx + int(labels[j].index(df[i, j]))] = 1
                    # one hot encoding is of length of the list of labels
                    idx += len(labels[j])
                # in each row of the matrix the all 1 hot encodings in that row are combined horizontally

                # gender is protected attribute
                # finds gender feature
                elif 'ale' in labels[j][0]:
                    for i in range(df.shape[0]):
                        # not sure why converted to int since it's already an int
                        u[i] = int(labels[j].index(df[i, j]))

                # binary features excluding gender
                # only option is greater or less than 50K
                else:
                    for i in range(df.shape[0]):
                        y[i] = int(labels[j].index(df[i, j]))
            else:
                # 1 if datapoint is above median, 0 otherwise
                # numeric features added on, each binary, whether or not above the median
                for i in range(df.shape[0]):
                    d[i, idx] = float(df[i, j] > labels[j])
                idx += 1
        return d.astype(np.bool), u.astype(np.bool), y.astype(np.bool)  # observation, protected, label

    adult_binary = transform_to_binary(adult_data, adult_labels)
    adult_test_binary = transform_to_binary(adult_test, adult_labels)

    # export as binary pickle file
    # wb - write binary
    with open('adult_binary.pkl', 'wb') as f:
        pkl.dump(adult_binary, f)
    with open('adult_test_binary.pkl', 'wb') as f:
        pkl.dump(adult_test_binary, f)


# function used for both adult and mental health dataset
def create_adult_or_mh_datasets(batch=64):
    # adult_bool is True for adult dataset, False for mental health
    from lag_fairness.examples.adult import adult_bool

    if adult_bool:
        # rb - read binary
        with open('adult_binary.pkl', 'rb') as f:
            ab = pkl.load(f)
        with open('adult_test_binary.pkl', 'rb') as f:
            atb = pkl.load(f)
    else:
        # rb - read binary
        with open('mh_binary_train.pkl', 'rb') as f:
            ab = pkl.load(f)
        with open('mh_binary_test.pkl', 'rb') as f:
            atb = pkl.load(f)

    # contains 3 arrays corresponding to d, u , y
    # 30162 data points
    # tuple - exactly like list except immutable
    # three arrays - d, u, y  - observations, sensitive, gender
    train = [a.astype(np.float32) for a in ab]
    test = [a.astype(np.float32) for a in atb]

    # arrays need to be reshaped for mental health dataset
    if not adult_bool:
        train[1] = np.reshape(train[1], (-1,1))
        test[1] = np.reshape(test[1], (-1, 1))
        train[2] = np.reshape(train[2], (-1,1))
        test[2] = np.reshape(test[2], (-1, 1))

    adult_binary = tuple(train)
    adult_test_binary = tuple(test)

    # prefetch 64 batches of 64 elements
    # https: // www.tensorflow.org / api_docs / python / tf / data / Dataset  # prefetch
    # from_tensor_slices emits data 1 element at a time
    # shuffle entire dataset then split into batches
    # .shuffle() takes in buffer_size, should be greater or equal to size of the dataset
    # prefetch improves performance
    train = tf.data.Dataset.from_tensor_slices(adult_binary).shuffle(adult_binary[0].shape[0]).batch(batch).prefetch(batch)
    test = tf.data.Dataset.from_tensor_slices(adult_test_binary).batch(batch).prefetch(batch)

    # p(u) is the distibution of the binary sensitive gender attribute
    # probs is probability of the positive event of male
    # p = 0.67568463 adult, 0.7533 for mh
    # adult_binary[1] is sensitive variable u
    pu = tfd.Bernoulli(probs=np.mean(adult_binary[1]))
    return train, test, pu


if __name__ == '__main__':
    print('saving dataset')
    save_adult_datasets()
    #reate_adult_datasets()
