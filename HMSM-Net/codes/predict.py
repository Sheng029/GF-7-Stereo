import argparse
from hmsmnet import HMSMNet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("left_dir", type=str, help="directory path of left images")
    parser.add_argument("right_dir", type=str, help="directory path of right images")
    parser.add_argument("output_dir", type=str, help="directory path of predicted disparity maps")
    parser.add_argument("weights", type=str, help="path of trained weights")

    args = parser.parse_args()

    network = HMSMNet(1024, 1024, 1, -128.0, 64.0)
    network.buildModel()
    network.predict(args.left_dir, args.right_dir, args.output_dir, args.weights)
