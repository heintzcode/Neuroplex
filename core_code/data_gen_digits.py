import random
import re
import numpy as np

def create_digit_data(num_samples, window_size, complex_event_defs, mnist_rows_dict=None):
    """
    Create a "window" of events by creating a string of random numbers of window length
    Determine the location and class of any complex events in that window through regular expression matching
    -- complex_event_defs are a list of compiled regexes
    Turn each individual event into a onehot array
    Concatenate the "window" of onehots and pair with its labels
    If mnist, associate each event with a relevant row from the training set, and return the index of that row
    """
    window_data = []
    label_counts = []
    image_indices = []
    for i in range(num_samples):
        sample = "".join([str(random.choice(range(10))) for j in range(window_size)])
        label_count = []
        for j in range(len(complex_event_defs)):
            matches = complex_event_defs[j].findall(sample)
            label_count.append(len(matches))
        onehots = sample_to_onehots(sample)
        window_data.append(onehots)
        label_counts.append(label_count)

        if mnist_rows_dict is not None:
            images = []
            for s in sample:
                images.append(random.choice(mnist_rows_dict[int(s)]))
            image_indices.append(np.array(images))
    return np.array(window_data), np.array(label_counts), np.array(image_indices)

def sample_to_onehots(samplestring, num_event_types=10):
    """
    The sample has n events (length n)
    Each is described as a one-hot vector of length m, where m is the number of event types (10 digits)
    So the 10-digit string becomes 10x10 array
    """
    result = []
    for s in samplestring:
        onehot_event = np.zeros(num_event_types)
        onehot_event[int(s)] = 1
        result.append(np.array(onehot_event))
    return np.array(result)

def load_mnist_data():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    complex_event_strings = ["1.*2.*3", "4.*5.*6", "7.*8", "9.*0"]
    complex_event_defs = [re.compile(ce_string) for ce_string in complex_event_strings]
    data, label_counts, images = create_digit_data(50, 10, complex_event_defs)
    for i in range(50):
        print(data[i], label_counts[i])
        # print("{}; {}".format(data[i], ",".join([complex_event_strings[j] for j in range(len(complex_event_defs)) if label_counts[i][j]>0 ])))
