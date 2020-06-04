import argparse
import shutil
import os

def main(args):
    dirs = ["train", "val", "test"]
    for dr in dirs:
        # first copy records
        print("Copying records from src to dst for dir {}".format(dr))
        s = open("{}/{}_records.txt".format(args.src, dr), "r")
        t = open("{}/{}_records.txt".format(args.dst, dr), "a")
        lines = s.readlines()
        t.writelines(lines)

        #next move files
        print("Moving files from {} to {}".format(args.src, args.dst))
        src_files = os.listdir("{}/{}".format(args.src, dr))
        src_path = "{}/{}".format(args.src, dr)
        dst_path = "{}/{}".format(args.dst, dr)
        for f in src_files:
            shutil.move(os.path.join(src_path, f), os.path.join(dst_path, f))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',
                        type=str,
                        required=True,
                        help='Dataset to merge. This dataset will be merged with the destination folder. Provide a fully qualified path.')
    parser.add_argument('--dst',
                        type=str,
                        required=True,
                        help='This folder will be retained and will have final merged contents. Provide a fully qualified path.')

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    main(args)

