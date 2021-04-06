# plain python
import os
import random
import glob
from types import SimpleNamespace

# libs
import click
from datasets import load_dataset
import torch
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    AutoConfig,
)
import wandb

# local
from utils import Preprocessor
from roberta_bilstm import RobertaBiLSTMForTokenClassification

click_datasets = click.Choice(
    [
        "iwslt2011-validation",
        "iwslt2011-test",
        "iwslt2011-train",
        "mgb-train",
        "mgb-validation",
    ],
    case_sensitive=False,
)


@click.command()
@click.option("--run-name", type=str)
@click.option("--model-name", type=str)
@click.option("--weight-decay", type=float)
@click.option("--lr", type=float)
@click.option("--batch-size", type=int)
@click.option("--gradient-accumulation", type=int)
@click.option("--epochs", type=int)
@click.option(
    "--resample",
    type=click.Choice(["None", "Max", "Mean", "Median"], case_sensitive=False),
)
@click.option("--max-length", type=int)
@click.option("--log-steps", type=int)
@click.option("--lookahead", nargs=2, type=click.Tuple([int, int]))
@click.option("--truncate-left", is_flag=True, default=False)
@click.option("--include-pauses", is_flag=True, default=False)
@click.option("--replace-pause", default=None, type=str)
@click.option("--pause-threshold", default=0.2, type=float)
@click.option("--no-train", is_flag=True, default=False)
@click.option("--teacher-forcing", is_flag=True, default=False)
@click.option("--tagging", is_flag=True, default=False)
@click.option("--dryrun", is_flag=True, default=False)
@click.option("--bilstm", is_flag=True, default=False)
@click.option("--save", type=str)
@click.option("--load", type=str)
@click.option("--num-proc", type=int)
@click.option("--train-dataset", type=click_datasets)
@click.option("--validation-dataset", type=click_datasets)
def train(**kwargs):
    kwargs["real_batch_size"] = kwargs["batch_size"] * kwargs["gradient_accumulation"]
    kwargs["train_dataset"] = kwargs["train_dataset"].split("-")
    kwargs["validation_dataset"] = kwargs["validation_dataset"].split("-")
    args = SimpleNamespace(**kwargs)
    if args.dryrun:
        os.environ['WANDB_MODE'] = 'dryrun'
    run = wandb.init(project="SAPAUT-PAUSES", name=args.run_name)
    if args.load:
        if not os.path.exists(f"models/{args.load}"):
            artifact = run.use_artifact(args.load + ":latest")
            artifact.download(root=f"models/{args.load}")
        args.model_name = f"models/{args.load}"
    special_tokens = ["<punct>"]
    if args.include_pauses:
        if not args.replace_pause:
            special_tokens.append("<pause>")
        ds_type = "ref-pauses"
    else:
        ds_type = "ref"
    if args.teacher_forcing:
        ds_type += "-tf"
    if args.tagging:
        ds_type += "-tag"
    if args.pause_threshold != 0.2:
        download_mode="reuse_cache_if_exists"
    else:
        download_mode="reuse_dataset_if_exists"
    ds_train = load_dataset(
        f"punctuation-iwslt2011/{args.train_dataset[0]}.py",
        ds_type,
        download_mode=download_mode,
        splits=[args.train_dataset[1]],
        ignore_verifications=True,
        lookahead_range=args.lookahead,
        pause_threshold=args.pause_threshold,
    )
    print("len", len(ds_train["validation"]))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, fast=True, additional_special_tokens=special_tokens, add_prefix_space=args.tagging
    )
    if not args.tagging:
        label_names = ds_train[args.train_dataset[1]].features["label"].names
    else:
        label_names = ds_train[args.train_dataset[1]].features["label"].feature.names
    preprocessor = Preprocessor(
        tokenizer,
        args,
        label_names,
        args.replace_pause,
        args.tagging,
    )
    ds_train = ds_train.map(
        preprocessor.preprocess, batched=False, num_proc=args.num_proc
    )
    ds_train.rename_column_("label", "labels")
    ds_valid = load_dataset(
        f"punctuation-iwslt2011/{args.validation_dataset[0]}.py",
        ds_type,
        download_mode=download_mode,
        splits=[args.validation_dataset[1]],
        ignore_verifications=True,
        lookahead_range=args.lookahead,
        pause_threshold=args.pause_threshold,
    )
    ds_valid = ds_valid.map(
        preprocessor.preprocess, batched=False, num_proc=args.num_proc
    )
    ds_valid.rename_column_("label", "labels")
    train = ds_train[args.train_dataset[1]]
    valid = ds_valid[args.validation_dataset[1]]
    train.shuffle(42)
    valid.shuffle(42)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        logging_dir="./logs",
        logging_steps=args.log_steps,
        evaluation_strategy="steps",
        gradient_accumulation_steps=args.gradient_accumulation,
        eval_steps=args.log_steps,
    )

    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=4,
    )

    if not args.tagging:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, config=config
        )
    else:
        if args.bilstm:
            model = RobertaBiLSTMForTokenClassification.from_pretrained(
                args.model_name, config=config
            )
        else:
            model = AutoModelForTokenClassification.from_pretrained(
                args.model_name, config=config
            )
    model.resize_token_embeddings(len(tokenizer))

    optimizer = AdamW(
        [
            {"params": model.base_model.parameters()},
            {"params": model.classifier.parameters()},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.resample != "None":
        function_dict = {
            "Max": np.max,
            "Mean": np.mean,
            "Median": np.median,
        }
        np_function = function_dict[args.resample]
        mean_samples_excl_none = int(
            np_function(sorted(np.unique(train["labels"], return_counts=True)[1])[:-1])
        )
        per_class_samples = mean_samples_excl_none
        balanced_filter = np.concatenate(
            [
                np.where(np.array(train["labels"]) == i)[0][:per_class_samples]
                for i in range(4)
            ],
            axis=0,
        )
        train = train.select(balanced_filter)

    total_steps = len(train) // args.real_batch_size
    total_steps = total_steps * args.epochs
    schedule = get_linear_schedule_with_warmup(optimizer, total_steps // 2, total_steps)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        compute_metrics=preprocessor.compute_metrics,
        optimizers=(optimizer, schedule),
    )

    wandb.config.update(args.__dict__)

    if not args.no_train:
        trainer.train()

    if args.tagging:
        return
    
    lookahead_test = []
    for i in range(5):
        lookahead_test.append(
            valid.select(np.where(np.array(valid["lookahead"]) == i)[0])
        )
    la_metrics = []
    for l_test in lookahead_test:
        la_metrics.append(trainer.predict(l_test).metrics)

    for k in la_metrics[0].keys():
        data = [[i, m[k]] for i, m in enumerate(la_metrics)]
        table = wandb.Table(data=data, columns=["lookahead", k])
        wandb.log(
            {
                f"{k}_lookahead": wandb.plot.line(
                    table, "lookahead", k, title=f"{k} vs. lookahead"
                )
            }
        )

    if args.save:
        trainer.save_model(f"models/{args.save}")
        tokenizer.save_pretrained(f"models/{args.save}")
        model_artifact = wandb.Artifact(args.save, type="model")
        for path in glob.glob(f"models/{args.save}/**/*.*", recursive=True):
            model_artifact.add_file(path)
        wandb.run.log_artifact(model_artifact)

    for i in range(5):
        res_dict = {key: round(val * 100, 1) for key, val in la_metrics[i].items()}
        print(f"------- {i} ----------")
        print(
            "COMMA",
            res_dict["eval_precision_<comma>"],
            res_dict["eval_recall_<comma>"],
            res_dict["eval_f1_<comma>"],
        )
        print(
            "PERIOD",
            res_dict["eval_precision_<period>"],
            res_dict["eval_recall_<period>"],
            res_dict["eval_f1_<period>"],
        )
        print(
            "QUESTION",
            res_dict["eval_precision_<question>"],
            res_dict["eval_recall_<question>"],
            res_dict["eval_f1_<question>"],
        )
        print(
            "OVERALL",
            res_dict["eval_precision"],
            res_dict["eval_recall"],
            res_dict["eval_f1"],
        )
        print()


if __name__ == "__main__":
    train()
