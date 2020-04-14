
import model

########################GLOBAL########################################
args = parser.parse_args()
print(args)

# predefining random initial seeds
th.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

visual_feat_path = r"C:\Users\dcsang\PycharmProjects\embedding\breakfast\Breakfast_fs\data_maxpool_splits\split4"
text_path = r"C:\Users\dcsang\PycharmProjects\embedding\breakfast\Breakfast_fs\groundTruth_maxpool_clean_splits\split4"
map_path = r"C:\Users\dcsang\PycharmProjects\embedding\breakfast\Breakfast_fs\splits\mapping_clean.txt"
log_path = r"C:\Users\dcsang\PycharmProjects\embedding\MEE-BF-SingleSrc\logs"

device = #TODO
########################GLOBAL########################################

def get_accuracy(groundtruth, prediction):
    groundtruth = np.array(groundtruth)
    prediction = np.array(prediction)
    return np.sum(prediction == groundtruth) / float(len(groundtruth))


def get_pred_idx(sim_matrix):
    #matrix should be uniq_count x BS. check. calculate argmin accordingly

running_loss_iter=0
def train(net,optimizer, max_margin, train_dataloader, mod_name, uniq_w2v, w2v_type, epoch, writer):
    print("Starting Training Loop for Epoch: "+ str(epoch+1))
    net.train()

    gt = []
    pred = []
    running_loss = 0.0

    uniq_w2v_th = th.Tensor(uniq_w2v[w2v_type]).to(device)
    for i_batch, sample_batched in enumerate(train_dataloader):
        mod = {}
        mod_ind = {}
        for src in mod_name:
            mod[src] = th.Tensor(sample_batched[src]).to(device)
            mod_ind[src] = th.Tensor(sample_batched[src+"_ind"]).to(device)

        #TODO: stereo02?

        optimizer.zero_grad()

        similarity_matrix = net(mod, mod_ind, uniq_w2v_th) #TODO #TODO: verify dims: uniq_count x Batch size
        curr_pred_idx = get_pred_idx(similarity_matrix.numpy())
        curr_gt_idx = sample_batched["label_idx"]
        gt.extend(curr_gt_idx.tolist())
        pred.extend(curr_pred_idx.tolist())

        loss = max_margin() #TODO: need to reimplement this loss fn. similarity matrix needs to be transposed,
        loss.backward()
        optimizer.step()
        running_loss += loss.data

        if (i_batch+1)%args.n_display ==0:
            print('Epoch %d, Epoch status: %.6f, Training loss: %.4f' % (epoch + 1,
                                                                         args.train_batch_size * float(
                                                                             i_batch) / len(train_dataset),
                                                                         running_loss / n_display))
            running_loss_iter += 1
            writer.add_scalar('Running loss', running_loss, running_loss_iter)
            running_loss = 0.0

    accuracy = get_accuracy(gt,pred)
    print("Epoch %d, Training Accuracy: %.7f" % (epoch + 1, accuracy))
    writer.add_scalar("Training Accuracy per epoch", accuracy * 100, epoch + 1)

def test():
    net.eval()


def main():
    #TODO: Hyper-param tuning code here. Ensure this block is before utils.print_hyperparams() fn call
    #TODO: in hyperparameter tuning code, make sure you change val of args.X so that correct values printed in tensorboard with print_hyperparams fn

    train_dataset = #TODO
    train_dataloader = #TODO

    test_dataset = #TODO
    test_dataloader = #TODO

    w2v_type = "num_w2v" #or text_w2v
    #train_uniq_w2v =
    #test_uniq_w2v =

    print("Dataset Loaded")
    print("Size of training", len(train_dataset))
    print("Size of testing", len(test_dataset))

    log_dir = str(int(time.time()))
    writer = SummaryWriter(os.path.join(log_path, log_dir))
    utils.print_hyperparams_tb(writer, args)

    #Setup
    w2v_dim = 200
    mod_dim = train_dataset.mod_dim
    assert train_dataset.mod_dim == test_dataset.mod_dim

    net = model.Net(mod_dim, w2v_dim, args.latent_feat_dim).to(device)
    max_margin = MaxMarginRankingLoss(margin=args.margin).to(device)
    if args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    #TODO: define scheduler here

    for epoch in range(args.epoch):
# sim matrix here is transpose of original code. beware
#learning rate scheduler
