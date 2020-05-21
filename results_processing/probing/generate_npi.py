import argparse
import jsonlines
import math
import random
import pdb

from sklearn.model_selection import train_test_split


def reformat(pair, split, rate, i, template):
    return {
        "sentence_good": pair["sentence_good"],
        "sentence_bad": pair["sentence_bad"],
        "template": template,
        "UID": f"{template}-{i}-{pair['UID']}-{split}-{rate}",
        "negated_outside_scope": pair["negated_outside_scope"] == 'yes'
    }

def split(pair):
    good = {
        "sentence": pair["sentence_good"],
        "template": pair["template"],
        "UID": pair["UID"],
        "label": "good",
        "co-occurs": True,
        "negated_outside_scope": pair["negated_outside_scope"]
    }
    bad = {
        "sentence": pair["sentence_bad"],
        "template": pair["template"],
        "UID": pair["UID"],
        "label": "bad",
        "co-occurs": not pair["negated_outside_scope"],
        "negated_outside_scope": pair["negated_outside_scope"]
    }
    return (good, bad)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rate",
        default=1.0,
        type=float,
        help="The rate of cooccurence between the good and the bad feature.",
    )
    parser.add_argument(
        "--split",
        default="sentential_negation_npi_scope_probing",
        type=str,
        help="Input data split.",
    )
    parser.add_argument("--test_size", default=1500, type=int, help="Test size.")
    parser.add_argument("--train_size", default=2000, type=int, help="Train size.")
    parser.add_argument("--val_size", default=500, type=int, help="Val size.")
    args = parser.parse_args()

    # TODO: clean this up
    data_A = []
    for i in range(1, 6):
        with jsonlines.open("./outputs/benchmark/sentential_negation_npi_scope_all_{}.jsonl".format(i)) as reader:
            new = [obj for obj in reader if "never" not in obj['sentence_good']]
            new = [reformat(d, args.split, args.rate, i, "A") for d in new]
        data_A += new
    random.shuffle(data_A)
    
    data_B = []
    for i in range(1, 6):
        with jsonlines.open("./outputs/benchmark/sentential_negation_npi_scope_B_{}.jsonl".format(i)) as reader:
            new = [obj for obj in reader]
            new = [reformat(d, args.split, args.rate, i, "B") for d in new]
        data_B += new
    random.shuffle(data_B)

    co_occur_A = []
    not_co_occur_A = []
    for data in data_A:
        (good, bad) = split(data)
        if data["negated_outside_scope"]:
            co_occur_A.append(good)
            not_co_occur_A.append(bad)
        else:
            co_occur_A.append(good)
            co_occur_A.append(bad)

    co_occur_B = []
    not_co_occur_B = []
    for data in data_B:
        (good, bad) = split(data)
        if data["negated_outside_scope"]:
            co_occur_B.append(good)
            not_co_occur_B.append(bad)
        else:
            co_occur_B.append(good)
            co_occur_B.append(bad)
        
    # Making these the same length, so there's an equal number from each template
    co_occur = co_occur_A[:1500] + co_occur_B[:1500]
    random.shuffle(co_occur)

    not_co_occur = not_co_occur_A[:1000] + not_co_occur_B[:1000]
    random.shuffle(not_co_occur)

    test_data = co_occur[: args.test_size // 2] + not_co_occur[: args.test_size // 2]
    train_data_co_occur = co_occur[args.test_size // 2 : ]
    train_data_not_co_occur = not_co_occur[args.test_size // 2 :]

    train_data = (
        train_data_co_occur[: math.floor(args.train_size * args.rate)]
        + train_data_not_co_occur[: math.ceil(args.train_size * (1 - args.rate))]
    )
    val_data_co_occur = train_data_co_occur[math.floor(args.train_size * args.rate) :]
    val_data_not_co_occur = train_data_not_co_occur[math.ceil(args.train_size * (1 - args.rate)) :]
    val_data = (
        val_data_co_occur[: math.floor(args.val_size * args.rate)]
        + val_data_not_co_occur[: math.ceil(args.val_size * (1 - args.rate))]
    )

    print(
        len(val_data),
        len([d["co-occurs"] for d in val_data if d["co-occurs"]]) / len(val_data),
    )
    print(
        len(train_data),
        len([d["co-occurs"] for d in train_data if d["co-occurs"]]) / len(train_data),
    )
    print(
        len(test_data),
        len([d["co-occurs"] for d in test_data if d["co-occurs"]]) / len(test_data),
    )

    with jsonlines.open(
        f"./outputs/probing/val-{args.split}-{args.rate}.jsonl", mode="w"
    ) as writer:
        writer.write_all(val_data)

    with jsonlines.open(
        f"./outputs/probing/train-{args.split}-{args.rate}.jsonl", mode="w"
    ) as writer:
        writer.write_all(train_data)

    # We test on the same data across different rates.
    with jsonlines.open(
        f"./outputs/probing/test-{args.split}.jsonl", mode="w"
    ) as writer:
        writer.write_all(test_data)
