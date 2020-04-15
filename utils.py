from sklearn.metrics import confusion_matrix
import pandas as pd


def print_hyperparams_tb(writer, args, train_dataset_size, test_dataset_size):
    writer.add_text('Dataset', 'Training set size: ' + str(train_dataset_size))
    writer.add_text('Dataset', 'Test set size: ' + str(test_dataset_size))

    writer.add_text('Hyperparams', 'epochs: ' + str(args.epochs))
    writer.add_text('Hyperparams', 'batch size: ' + str(args.train_batch_size))
    writer.add_text('Hyperparams', 'learning rate: ' + str(args.lr))
    writer.add_text('Hyperparams', 'margin: ' + str(args.margin))
    writer.add_text('Hyperparams', 'optimizer: ' + str(args.optimizer))

def map_back(lst, dct):
    output = [dct[i] for i in lst]
    return output

def get_confusion_matrix(groundtruth, prediction, stoi_map, itos_map):
    labels = list(stoi_map.keys())
    labels.remove("SIL")
    cm = confusion_matrix(map_back(groundtruth, itos_map), map_back(prediction, itos_map), labels=labels)
    cm_pd = pd.DataFrame(cm, index=labels, columns=labels)
    return cm_pd
