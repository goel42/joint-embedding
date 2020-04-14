import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--latent_feat_dim', type = int, default=53,
                    help = "dimensionality of the latent space")
parser.add_argument('--vis_feat_dim', type=int, default=400,
                    help="dimensionality of video frame features")
parser.add_argument('--rm_bkg_frames', type=bool, default=True,
                    help="remove background frames(SIL) from the train and test data")
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=300,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--test_batch_size', type=int, default=1000,
                    help='batch size')
parser.add_argument('--margin', type=float, default=0.30000000000000004,
                    help='MaxMargin margin value')
parser.add_argument('--lr_decay', type=float, default=0.95,
                    help='Learning rate exp epoch decay')
parser.add_argument('--n_display', type=int, default=5,
                    help='Information display frequency')
parser.add_argument('--GPU', type=bool, default=True,
                    help='Use of GPU')
parser.add_argument('--n_cpu', type=int, default=1,
                    help='Number of CPU')
parser.add_argument('--seed', type=int, default=1,
                    help='Initial Random Seed')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='optimizer')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Nesterov Momentum for SGD')