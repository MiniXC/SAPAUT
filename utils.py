import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


class Preprocessor:
    def __init__(self, tokenizer, args, feature_names, replace_pause=None, tagging=False):
        self.tokenizer = tokenizer
        self.p = args
        self.features = feature_names
        self.replace_pause = replace_pause
        self.tagging = tagging

    def encode_tags(self, tags, encodings):
        words = encodings.words()
        doc_enc_labels = np.ones(len(encodings["offset_mapping"][0]),dtype=int) * -100
        count = 0
        for i, word_id in enumerate(words):
            if word_id is not None and (i == 0 or words[i-1] != word_id):
                doc_enc_labels[i] = tags[count]
                count += 1
        return doc_enc_labels.tolist()
        
    def preprocess(self, e):
        if self.replace_pause is not None:
            if self.tagging:
                e["tokens"] = [t.replace("<pause>", self.replace_pause) for t in e["tokens"]]
            else:
                e["text"] = e["text"].replace("<pause>", self.replace_pause)
        if self.tagging:
            to_tokenize = e["tokens"][-self.p.max_length:]
        else:
            to_tokenize = e["text"]
        result = self.tokenizer(
            to_tokenize,
            padding=True,
            max_length=self.p.max_length,
            pad_to_multiple_of=self.p.max_length,
            truncation=(not self.p.truncate_left and not self.tagging),
            return_tensors="pt",
            is_split_into_words=self.tagging,
            return_offsets_mapping=self.tagging,
        )
        if self.tagging:
            result["label"] = self.encode_tags(
                e["label"][-self.p.max_length:],
                result,
            )
        if len(result["input_ids"][0]) > self.p.max_length:
            if self.tagging:
                result["label"] = np.concatenate(
                [
                    [-100],
                    result["label"][
                        1 : np.where(
                            result["input_ids"][0] == self.tokenizer.eos_token_id
                        )[0][0]
                    ][-(self.p.max_length - 2) :],
                    [-100],
                ]
            )
            result["input_ids"] = np.concatenate(
                [
                    [self.tokenizer.bos_token_id],
                    result["input_ids"][0][
                        1 : np.where(
                            result["input_ids"][0] == self.tokenizer.eos_token_id
                        )[0][0]
                    ][-(self.p.max_length - 2) :],
                    [self.tokenizer.eos_token_id],
                ]
            )
            result["attention_mask"] = result["attention_mask"][0][: self.p.max_length]
        else:
            result["input_ids"] = result["input_ids"][0]
            result["attention_mask"] = result["attention_mask"][0]
        if not self.tagging:
            result["lookahead"] = e["lookahead"]
        return result

    def compute_metrics(self, eval_pred):
        ignore_index = self.features.index("<none>")
        predictions, labels = eval_pred
        metrics = {}
        if self.tagging:
            predictions_orig, labels_orig = predictions, labels
            for i in range(34):
                new_p = []
                new_l = []
                for p, l in zip(predictions_orig, labels_orig):
                    if len(l) > i+1 and l[i] != -100:
                        new_p.append(p[i])
                        new_l.append(l[i])
                predictions = new_p
                labels = new_l
                if len(new_l) > 0:
                    predictions = np.argmax(predictions, axis=1)
                    num_punct_true = len([l for l in labels if l != ignore_index])
                    num_punct_pred = len([p for p in predictions if p != ignore_index])
                    num_punct_correct = np.sum(
                        [p == l for p, l in zip(labels, predictions) if l != ignore_index]
                    )
                    if "precision" not in metrics:
                        metrics["precision"] = []
                    if "recall" not in metrics:
                        metrics["recall"] = []
                    if "f1" not in metrics:
                        metrics["f1"] = []
                    metrics["precision"].append(num_punct_correct / num_punct_pred)
                    metrics["recall"].append(num_punct_correct / num_punct_true)
                    metrics["f1"].append((2 * metrics["precision"][-1] * metrics["recall"][-1]) / (
                        metrics["precision"][-1] + metrics["recall"][-1]
                    ))
                    for name, f in zip(self.features, f1_score(labels, predictions, average=None)):
                        if f"f1_{name}" not in metrics:
                            metrics[f"f1_{name}"] = []
                        metrics[f"f1_{name}"].append(f)
                    for name, f in zip(
                        self.features, precision_score(labels, predictions, average=None)
                    ):
                        if f"precision_{name}" not in metrics:
                            metrics[f"precision_{name}"] = []
                        metrics[f"precision_{name}"].append(f)
                    for name, f in zip(
                        self.features, recall_score(labels, predictions, average=None)
                    ):
                        if f"recall_{name}" not in metrics:
                            metrics[f"recall_{name}"] = []
                        metrics[f"recall_{name}"].append(f)
            return metrics
        else:
            predictions = np.argmax(predictions, axis=1)
            num_punct_true = len([l for l in labels if l != ignore_index])
            num_punct_pred = len([p for p in predictions if p != ignore_index])
            num_punct_correct = np.sum(
                [p == l for p, l in zip(labels, predictions) if l != ignore_index]
            )
            metrics = {
                "precision": num_punct_correct / num_punct_pred,
                "recall": num_punct_correct / num_punct_true,
            }
            metrics["f1"] = (2 * metrics["precision"] * metrics["recall"]) / (
                metrics["precision"] + metrics["recall"]
            )
            for name, f in zip(self.features, f1_score(labels, predictions, average=None)):
                metrics[f"f1_{name}"] = f
            for name, f in zip(
                self.features, precision_score(labels, predictions, average=None)
            ):
                metrics[f"precision_{name}"] = f
            for name, f in zip(
                self.features, recall_score(labels, predictions, average=None)
            ):
                metrics[f"recall_{name}"] = f
            return metrics
