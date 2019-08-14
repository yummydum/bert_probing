"""
For Hydrogen;
%load_ext autoreload
%autoreload 2
"""

import argparse
from pathlib import Path
import json
import logging

def main(args):

    # Init the dicts
    result_dict = dict()
    result_dict["max_sentence_len"] = 0
    result_dict["data"] = dict()
    for i in ["all_pos","all_cpos","all_relation"]:
        result_dict[i] = set()
    entry_dict = dict()
    for key in ["position","word","cpos","pos","head","relation"]:
        entry_dict[key] = []

    count = 0
    conll_path = Path(f"probing_data/ST/en-universal-{args.which}.conll")
    with conll_path.open(mode="r",encoding="utf-8") as f: # f = conll_path.open(), f.close()
        for line in f:   # line = next(f)
            tok = line.strip().split('\t')
            logging.debug(tok)
            # If empty line, write the result
            if not tok or line.strip() == '':
                logging.debug(tok)
                logging.debug(entry_dict)
                if len(entry_dict["word"])>1:
                    result_dict["data"][count] = entry_dict.copy()
                    count += 1
                    if len(entry_dict["word"]) > result_dict["max_sentence_len"]:
                        result_dict["max_sentence_len"] = len(entry_dict["word"])
                # Init entry dict
                for key in entry_dict.keys():
                    entry_dict[key] = []
                # Report progress
                if count >= 5000 and count % 5000 == 0:
                    logging.info(f"Now at {count}th example")
                if args.debug and count > 5:
                    break
            # else continue constructing the entry list
            else:
                entry_dict["position"].append(int(tok[0]))
                entry_dict["word"].append(tok[1].lower())
                entry_dict["cpos"].append(tok[3].upper())
                entry_dict["pos"].append(tok[4].upper())
                entry_dict["cpos"].append(tok[5])
                entry_dict["head"].append(int(tok[6]))
                entry_dict["relation"].append(tok[7])
                # Track all pos/relation and max sentence len
                result_dict["all_cpos"].add(tok[3].upper())
                result_dict["all_pos"].add(tok[4].upper())
                result_dict["all_relation"].add(tok[7])

        # End of loop, if there is
        if len(entry_dict["word"]) > 1:
            result_dict["data"][count] = entry_dict.copy()
    logging.info(f"Process finished, total number of example is {count}")

    # Dump the result into json
    for i in ["all_pos","all_cpos","all_relation"]:
        result_dict[i] = list(result_dict[i])
    result_path = Path(f"probing_data/ST/ST-{args.which}.json")
    logging.info(f"Now writing the result to {result_path}")
    with result_path.open(mode="w") as f:
        json.dump(result_dict,f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("which",type=str,choices=["train","dev","test"],
                        help="Which data to process")
    parser.add_argument("--debug",action='store_true',
                        help="Debug mode if flagged")
    args = parser.parse_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level,
                        format='%(process)d-%(asctime)s-%(levelname)s-%(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')

    main(args)
