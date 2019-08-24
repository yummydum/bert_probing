"""
For Hydrogen;
%load_ext autoreload
%autoreload 2
"""

import argparse
from pathlib import Path
import csv
from util import set_logger


def main(args):

    for which in ["train", "dev", "test"]:  # which = "test"
        f_w = Path(f"data/ST/{which}.tsv").open("w")
        tsv_writer = csv.writer(f_w, delimiter="\t")
        f_r = Path(f"data/ST/en-universal-{which}.conll")

        # Init acc
        entries = ["position", "word", "cpos", "pos", "head", "relation"]
        entry_dict = dict()
        for e in entries:
            entry_dict[e] = []
        tsv_writer.writerow(entries)

        count = 0
        for line in f_r.open("r"):
            tok = line.strip().split('\t')
            # If empty line, write the result
            if not tok or line.strip() == '':
                if len(entry_dict["word"]) > 1:
                    row = []
                    for e in entries:
                        row.append(" ".join(entry_dict[e]))
                    tsv_writer.writerow(row)
                    count += 1

                # Init entry dict
                for key in entry_dict.keys():
                    entry_dict[key] = []
                # Report progress
                if count >= 5000 and count % 5000 == 0:
                    logger.info(f"Now at {count}th example")
                if args.debug and count == 5:
                    break
            # else continue constructing the entry list
            else:
                entry_dict["position"].append(tok[0])
                entry_dict["word"].append(tok[1].lower())
                entry_dict["cpos"].append(tok[3].upper())
                entry_dict["pos"].append(tok[4].upper())
                entry_dict["cpos"].append(tok[5])
                entry_dict["head"].append(tok[6])
                entry_dict["relation"].append(tok[7])

        # End of loop, if there is result write
        if len(entry_dict["word"]) > 1:
            row = []
            for e in entries:
                row.append(" ".join(entries[e]))
            tsv_writer.writerow(row)
        f_w.close()
        logger.info(f"Process finished, total number of example is {count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",
                        action='store_true',
                        help="Debug mode if flagged")
    args = parser.parse_args()
    # args = parser.parse_args(["--debug"])
    logger = set_logger(__name__)
    main(args)
