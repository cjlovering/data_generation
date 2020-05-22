import argparse
import jsonlines
import math
import random

from sklearn.model_selection import train_test_split


def reformat(pair, split, rate):
    return {
        "sentence_good": pair["sentence_good"],
        "sentence_bad": pair["sentence_bad"],
        "co-occurs": pair["object_plural"] == pair["subject_plural"],
        "UID": f"{pair['UID']}-{split}-{rate}",
    }


def sample(pair, rate):
    x = random.random()
    if x <= rate:
        # Keep this pair iff they co-occur.
        return pair["co-occurs"]
    else:
        # Keep this pair iff they do not co-occur.
        return not pair["co-occurs"]


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
        default="distractor_agreement_relational_noun_probing",
        type=str,
        help="Input data split.",
    )
    parser.add_argument("--test_size", default=1000, type=int, help="Test size.")
    parser.add_argument("--train_size", default=1000, type=int, help="Train size.")
    parser.add_argument("--val_size", default=500, type=int, help="Train size.")
    args = parser.parse_args()

    with jsonlines.open(f"./outputs/benchmark/{args.split}.jsonl") as reader:
        data = [obj for obj in reader]

    data = [reformat(d, args.split, args.rate) for d in data]
    random.shuffle(data)

    data_pos = [d for d in data if d["co-occurs"]]
    data_neg = [d for d in data if not d["co-occurs"]]

    test_data = data_pos[: args.test_size // 2] + data_neg[: args.test_size // 2]
    train_data_pos = data_pos[args.test_size // 2 :]
    train_data_neg = data_neg[args.test_size // 2 :]

    train_data = (
        train_data_pos[: math.floor(args.train_size * args.rate)]
        + train_data_neg[: math.ceil(args.train_size * (1 - args.rate))]
    )
    val_data_pos = train_data_pos[math.floor(args.train_size * args.rate) :]
    val_data_neg = train_data_neg[math.ceil(args.train_size * (1 - args.rate)) :]
    val_data = (
        val_data_pos[: math.floor(args.val_size * args.rate)]
        + val_data_neg[: math.ceil(args.val_size * (1 - args.rate))]
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
