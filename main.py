from argparse import ArgumentParser

from ner_spacy import train_ner_spacy, demo_spacy
from ner_transformers import run_model


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "model_type",
        choices=["spacy", "transformer"],
        help="Type of model to be trained/tested.",
    )
    parser.add_argument(
        "--train", action="store_true", help="Whether to train the model."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Whether to run a demo of the model."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="The path to the directory containing the training data.",
    )
    parser.add_argument(
        "--spacy_config", type=str, help="The path to the spaCy config file."
    )
    parser.add_argument(
        "--spacy_model", type=str, help="The path to the spaCy model directory."
    )
    parser.add_argument(
        "--transformer_model", type=str, help="The path to the transformer model."
    )
    return parser


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    if args.model_type == "spacy":
        if args.train:
            if not args.data_dir:
                raise ValueError("Please provide a data directory.")
            if not args.spacy_config:
                raise ValueError("Please provide a spaCy config file.")
            if not args.spacy_model:
                raise ValueError("Please provide a spaCy model directory.")
            train_ner_spacy(
                "spacy_config.cfg",
                args.spacy_model,
                create_data=True,
                data_dir=args.data_dir,
            )
        if args.demo:
            if not args.spacy_model:
                raise ValueError("Please provide a spaCy model directory.")
            demo_spacy(f"{args.spacy_model}/model-best")
    elif args.model_type == "transformer":
        if args.train and not args.data_dir:
            raise ValueError("Please provide a data directory.")
        if not args.transformer_model:
            raise ValueError("Please provide a transformer model path.")
        run_model(args.train, args.demo, args.data_dir, None, args.transformer_model)


if __name__ == "__main__":
    main()
