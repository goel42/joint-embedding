import model
import torch as th
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
from loss import MaxMarginRankingLoss
import os
import random
from torch.utils.tensorboard import SummaryWriter
from argparser import parser
import time
import sys
from Breakfast import Breakfast as BF
import utils

########################GLOBAL########################################
args = parser.parse_args()
print(args)

# predefining random initial seeds
th.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

visual_feat_path = r"C:\Users\dcsang\PycharmProjects\embedding\breakfast\Breakfast_fs\data_maxpool_splits\split1"
text_path = r"C:\Users\dcsang\PycharmProjects\embedding\breakfast\Breakfast_fs\groundTruth_maxpool_clean_splits\split1"
map_path = r"C:\Users\dcsang\PycharmProjects\embedding\breakfast\Breakfast_fs\splits\mapping_clean.txt"
log_path = r"C:\Users\dcsang\PycharmProjects\joint-embedding\logs"

device = th.device("cuda" if th.cuda.is_available() else "cpu")

running_loss_iter = 0


########################GLOBAL########################################

def get_accuracy(groundtruth, prediction):
    groundtruth = np.array(groundtruth)
    prediction = np.array(prediction)
    print("Length: ", len(groundtruth), len(prediction))
    # for g, p in zip(groundtruth, prediction):
    #     print(g, p)

    return np.sum(prediction == groundtruth) / float(len(groundtruth))


def train(net, optimizer, max_margin, train_dataloader, sources, labels_uniq_w2v, dataset_size, epoch, writer, stoi_map,
          itos_map):
    print("Starting Training Loop for Epoch: " + str(epoch + 1))
    net.train()

    gt = []
    pred = []
    running_loss = 0.0

    labels_uniq_w2v = th.Tensor(labels_uniq_w2v).to(device)
    for i_batch, sample_batched in enumerate(train_dataloader):
        mod = {}
        mod_ind = {}
        for src in sources:
            mod[src] = sample_batched[src].float().to(device)
            mod_ind[src] = sample_batched[src + "_ind"].float().to(device)
            # mask = mod_ind[src] != 0
            # mod[src] = mod[src][mask]
            # mod_ind[src] = mod_ind[src][mask]
        # TODO: stereo02?

        optimizer.zero_grad()

        similarity_matrix = net(mod, mod_ind, labels_uniq_w2v)  # TODO #TODO: verify dims: uniq_count x Batch size
        similarity_matrix = similarity_matrix.transpose(0,
                                                        1)  # potential bug #TODO: verify dims: Batch Size xuniq_count
        curr_pred_idx = similarity_matrix.argmax(axis=1)
        curr_gt_idx = sample_batched["labels_idx"]
        gt.extend(curr_gt_idx.tolist())
        pred.extend(curr_pred_idx.tolist())

        loss = max_margin(similarity_matrix, curr_gt_idx)
        # TODO: need to reimplement this loss fn. similarity matrix needs to be transposed,
        loss.backward()
        optimizer.step()
        running_loss += loss.data

        if (i_batch + 1) % args.n_display == 0:
            print('Epoch %d, Epoch status: %.6f, Training loss: %.4f' % (epoch + 1,
                                                                         args.train_batch_size * float(
                                                                             i_batch) / dataset_size,
                                                                         running_loss / args.n_display))
            global running_loss_iter
            running_loss_iter += 1
            writer.add_scalar('Running loss', running_loss, running_loss_iter)
            running_loss = 0.0
    accuracy = get_accuracy(gt, pred) * 100
    # cm = utils.get_confusion_matrix(gt, pred, stoi_map,itos_map)
    # print(cm)
    print("Epoch %d, Training Accuracy: %.7f" % (epoch + 1, accuracy))
    writer.add_scalar("Training Accuracy per epoch", accuracy, epoch + 1)
    return accuracy


def test(net, dataloader, sources, labels_uniq_w2v, dataset_size, epoch, writer, stoi_map, itos_map):
    print("Evaluating Epoch: %d" % (epoch + 1))
    net.eval()

    gt = []
    pred = []

    labels_uniq_w2v = th.Tensor(labels_uniq_w2v).to(device)
    for i_batch, sample_batched in enumerate(dataloader):
        mod = {}
        mod_ind = {}
        for src in sources:
            mod[src] = sample_batched[src].float().to(device)
            mod_ind[src] = sample_batched[src + "_ind"].float().to(device)

        similarity_matrix = net(mod, mod_ind, labels_uniq_w2v)  # TODO #TODO: verify dims: uniq_count x Batch size
        similarity_matrix = similarity_matrix.transpose(0,
                                                        1)  # potential bug #TODO: verify dims: Batch Size xuniq_count
        curr_pred_idx = similarity_matrix.argmax(axis=1)
        curr_gt_idx = sample_batched["labels_idx"]
        gt.extend(curr_gt_idx.tolist())
        pred.extend(curr_pred_idx.tolist())
        # DEBUG
        print(labels_uniq_w2v.requires_grad, similarity_matrix.requires_grad, )
    accuracy = get_accuracy(gt, pred) * 100
    print("Test Accuracy: %.7f" % accuracy)
    writer.add_scalar("Test Accuracy per epoch", accuracy, epoch + 1)
    return accuracy


def main():
    # TODO: Hyper-param tuning code here. Ensure this block is before utils.print_hyperparams() fn call TODO: in
    #  hyperparameter tuning code, make sure you change val of args.X so that correct values printed in tensorboard
    #  with print_hyperparams fn

    # sources =  ["cam01", "cam02", "webcam01", "webcam02", "stereo01", "stereo02"]
    sources = ["cam01", "cam02", "webcam01", "webcam02", "stereo01"]  # TODO: try with stereo02
    # sources = ["cam01"]
    activities = ['cereals', 'coffee', 'friedegg', 'juice', 'milk', 'pancake', 'salat', 'sandwich', 'scrambledegg',
                  'tea']
    train_persons = ["P" + str(num) for num in range(16, 54)]
    test_persons = ["P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10", "P11", "P12", "P13", "P14", "P15"]

    train_dataset = BF(os.path.join(visual_feat_path, "train"), os.path.join(text_path, "train"), map_path, sources,
                       activities, train_persons, rm_SIL=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)

    test_dataset = BF(os.path.join(visual_feat_path, "test"), os.path.join(text_path, "test"), map_path, sources,
                      activities, test_persons, rm_SIL=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

    # w2v of numbers or text?
    train_uniq_w2v = train_dataset.labels_uniq["labels_num_w2v"]
    test_uniq_w2v = test_dataset.labels_uniq["labels_num_w2v"]

    print("Dataset Loaded")
    train_dataset_size = len(train_dataset)
    print("Size of training", train_dataset_size)
    test_dataset_size = len(test_dataset)
    print("Size of testing", len(test_dataset))

    log_dir = str(int(time.time()))
    writer = SummaryWriter(os.path.join(log_path, log_dir))
    utils.print_hyperparams_tb(writer, args, train_dataset_size, test_dataset_size)

    # Setup
    w2v_dim = 200
    mod_dim = train_dataset.mod_dim
    assert train_dataset.mod_dim == test_dataset.mod_dim

    net = model.Net(mod_dim, w2v_dim, args.latent_feat_dim, device).to(device)
    max_margin = MaxMarginRankingLoss(margin=args.margin).to(device)
    if args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    # TODO: define scheduler here

    best_train_accu, best_test_accu, best_epoch = -np.inf, -np.inf, -np.inf
    for epoch in range(args.epochs):
        train_accu = train(net, optimizer, max_margin, train_dataloader, sources, train_uniq_w2v, train_dataset_size,
                           epoch, writer, train_dataset.stoi_map, train_dataset.itos_map)
        # print(list(net.parameters()))
        test_accu = test(net, test_dataloader, sources, test_uniq_w2v, test_dataset_size, epoch, writer,
                         test_dataset.stoi_map, test_dataset.itos_map)
        if test_accu > best_test_accu:
            best_train_accu, best_test_accu, best_epoch = train_accu, test_accu, epoch + 1

    print("Best Train Accuracy: %.7f Test Accuracy: %.7f (Epoch: %d)" % (best_train_accu, best_test_accu, best_epoch))


# sim matrix here is transpose of original code. beware
# learning rate scheduler

if __name__ == '__main__':
    main()
