import argparse
from hmsmnet import HMSMNet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("train_dir", type=str, help="directory path of training set")
    parser.add_argument("val_dir", type=str, help="directory path of training set")
    parser.add_argument("log_dir", type=str, help="directory path of training logs")
    parser.add_argument("weights", type=str, help="path to save weights")
    parser.add_argument("epochs", type=int, help="epochs of training")
    parser.add_argument("batch_size", type=int, help="batch size")

    args = parser.parse_args()

    network = HMSMNet(1024, 1024, 1, -128.0, 64.0)
    network.buildModel()
    network.train(args.train_dir, args.val_dir, args.log_dir, args.weights, args.epochs, args.batch_size)
