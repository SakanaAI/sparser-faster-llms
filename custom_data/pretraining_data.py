from itertools import chain
import contextlib
import logging
import torch
from datasets import load_dataset, disable_caching
from transformers import default_data_collator
from transformers.testing_utils import CaptureLogger
import evaluate
from transformers import PreTrainedTokenizerFast


class BlockDiagFromEOSFA2Collator:
    '''Custom collator returns the correct inputs for FlashAttention2 for masked
       LM training, removing access to past sequences separated by EOS'''

    def __init__(self, tokenizer, ignore_idx=-100, return_seq_idx=False):
        self.eos_id = tokenizer.eos_token_id
        self.ignore_idx = ignore_idx
        self.return_seq_idx = return_seq_idx

    def __call__(self, features):
        flat_ids, flat_labels, pos_ids = [], [], []
        seq_idx = [] if self.return_seq_idx else None
        cu = [0]
        max_len = 0

        for si, ex in enumerate(features):
            ids = ex["input_ids"]
            lbls = ex["labels"] if "labels" in ex else ids
            n = len(ids)
            if n == 0:
                continue

            start = 0
            split_on_eos = self.eos_id is not None
            for i, tok in enumerate(ids):
                if split_on_eos and tok == self.eos_id:
                    seg_ids = ids[start:i + 1]
                    seg_labels = lbls[start:i + 1]
                    if seg_ids:
                        seg_labels = [self.ignore_idx] + seg_labels[1:]
                        flat_ids += seg_ids
                        flat_labels += seg_labels
                        pos_ids += list(range(len(seg_ids)))
                        if self.return_seq_idx:
                            seq_idx += [si] * len(seg_ids)
                        cu.append(cu[-1] + len(seg_ids))
                        max_len = max(max_len, len(seg_ids))
                    start = i + 1

            if start < n:
                seg_ids = ids[start:n]
                if seg_ids:
                    seg_labels = lbls[start:n]
                    seg_labels = [self.ignore_idx] + seg_labels[1:]
                    flat_ids += seg_ids
                    flat_labels += seg_labels
                    pos_ids += list(range(len(seg_ids)))
                    if self.return_seq_idx:
                        seq_idx += [si] * len(seg_ids)
                    cu.append(cu[-1] + len(seg_ids))
                    max_len = max(max_len, len(seg_ids))

        assert cu[-1] == len(flat_ids) == len(flat_labels) == len(pos_ids)

        batch = {
            "input_ids": torch.tensor([flat_ids], dtype=torch.int64),
            "labels": torch.tensor([flat_labels], dtype=torch.int64),
            "position_ids": torch.tensor([pos_ids], dtype=torch.int64),
            "cu_seq_lens_q": torch.tensor(cu, dtype=torch.int32),
            "cu_seq_lens_k": torch.tensor(cu, dtype=torch.int32),
            "max_length_q": int(max_len),
            "max_length_k": int(max_len),
        }
        if self.return_seq_idx:
            batch["seq_idx"] = torch.tensor([seq_idx], dtype=torch.int32)
        return batch


class BlockDiagFromEOSCollator:
    def __init__(self, tokenizer):
        self.eos_id = tokenizer.eos_token_id

    def __call__(self, features):
        batch = default_data_collator(features)
        ids = batch["input_ids"]
        pad2d = batch.get("attention_mask")
        if pad2d is None:
            pad2d = torch.ones_like(ids)

        b, l = ids.shape
        dev = ids.device

        is_eos = ids.eq(self.eos_id)
        seg_start = torch.zeros_like(ids)
        seg_start[:, 0] = 1
        seg_start[:, 1:] = is_eos[:, :-1].int()
        seg_id = torch.cumsum(seg_start, dim=1) - 1

        same_seg = seg_id.unsqueeze(2) == seg_id.unsqueeze(1)
        t = torch.arange(l, device=dev)
        causal = t[None, :, None] >= t[None, None, :]
        keep = same_seg & causal
        keep = keep & pad2d[:, None, :].bool() & pad2d[:, :, None].bool()

        keep4 = keep.unsqueeze(1)
        attn = torch.zeros_like(keep4, dtype=torch.float32)
        attn.masked_fill_(~keep4, torch.finfo(attn.dtype).min)
        batch["attention_mask"] = attn

        labels = batch.get("labels")
        if labels is None:
            labels = ids.clone()
        else:
            labels = labels.clone()

        labels[pad2d.eq(0)] = -100
        ignore_next = torch.zeros_like(is_eos, dtype=torch.bool)
        ignore_next[:, 1:] = is_eos[:, :-1]
        labels[ignore_next] = -100
        batch["labels"] = labels

        return batch


logger = logging.getLogger(__name__)


def load_pretraining_dataset(
        tokenizer,
        max_seq_length,
        dataset_id_or_path=None,
        dataset_local_directory=None,
        dataset_config_name=None,
        load_dataset_kwargs=None,
        name=None,
        train_file=None,
        validation_file=None,
        keep_linebreaks=True,
        validation_split_percentage=None,
        streaming=False,
        preprocessing_num_workers=None,
        overwrite_cache=False,
        block_size=None,
        manually_add_eos=False,
        mask_past_sequences=False,
        max_train_samples=None,
        max_eval_samples=None,
        cache_dir=None,
        token=None,
        trust_remote_code=False,
        do_train=True,
        do_eval=True,
        skip_samples=None,
        attention_implementation=None,
        disable_caching_datasets=False,
        main_process_first=lambda desc: contextlib.nullcontext(),
):
    tok_log = logging.getLogger(
        "transformers.tokenization_utils_base"
    )

    if isinstance(tokenizer, PreTrainedTokenizerFast) and (
            not manually_add_eos):
        tok_log.warning(
            "For fast tokenizers, `manually_add_eos` must be True. Fast "
            "tokenizers are bugged and do not add the EOS token by default."
        )

    if disable_caching_datasets:
        disable_caching()
    if name is not None:
        if dataset_config_name is not None:
            assert dataset_config_name == name
        dataset_config_name = name

    if dataset_local_directory is not None:
        dataset_id_or_path = dataset_local_directory

    if load_dataset_kwargs is None:
        load_dataset_kwargs = {}
    if dataset_id_or_path is not None:
        raw = load_dataset(
            dataset_id_or_path,
            dataset_config_name,
            cache_dir=cache_dir,
            token=token,
            streaming=streaming,
            trust_remote_code=trust_remote_code,
            **load_dataset_kwargs,
        )
        if "validation" not in raw and validation_split_percentage is not None:
            pct = validation_split_percentage
            raw["validation"] = load_dataset(
                dataset_id_or_path,
                dataset_config_name,
                split=f"train[:{pct}%]",
                cache_dir=cache_dir,
                token=token,
                streaming=streaming,
                trust_remote_code=trust_remote_code,
                **load_dataset_kwargs,
            )
            raw["train"] = load_dataset(
                dataset_id_or_path,
                dataset_config_name,
                split=f"train[{pct}%:]",
                cache_dir=cache_dir,
                token=token,
                streaming=streaming,
                trust_remote_code=trust_remote_code,
                **load_dataset_kwargs,
            )
    else:
        files = {}
        if train_file:
            files["train"] = train_file
        if validation_file:
            files["validation"] = validation_file
        ext = (
            train_file.split(".")[-1] if train_file
            else validation_file.split(".")[-1]
        )
        if ext == "txt":
            ext, klb = "text", keep_linebreaks
        else:
            klb = None
        raw = load_dataset(
            ext,
            data_files=files,
            cache_dir=cache_dir,
            token=token,
            keep_linebreaks=klb,
            **load_dataset_kwargs,
        )
        if "validation" not in raw:
            pct = validation_split_percentage
            raw["validation"] = load_dataset(
                ext,
                data_files=files,
                split=f"train[:{pct}%]",
                cache_dir=cache_dir,
                token=token,
                **load_dataset_kwargs,
            )
            raw["train"] = load_dataset(
                ext,
                data_files=files,
                split=f"train[{pct}%:]",
                cache_dir=cache_dir,
                token=token,
                **load_dataset_kwargs,
            )

    cols = list(
        raw["train"].features if do_train
        else raw["validation"].features
    )
    text_col = "text" if "text" in cols else cols[0]

    if manually_add_eos:
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        def tok_fn(ex):
            with CaptureLogger(tok_log):
                return tokenizer(
                    ex[text_col],
                    add_special_tokens=False,
                    return_attention_mask=True,
                )
    else:
        def tok_fn(ex):
            with CaptureLogger(tok_log):
                return tokenizer(ex[text_col])

    with main_process_first("tokenization"):
        if not streaming:
            tok_ds = raw.map(
                tok_fn,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=cols,
                load_from_cache_file=not overwrite_cache,
            )
        else:
            tok_ds = raw.map(
                tok_fn,
                batched=True,
                remove_columns=cols,
            )

    if block_size is None:
        if tokenizer.model_max_length < max_seq_length:
            logger.warning(
                f"Tokenizer model max length ({tokenizer.model_max_length}) "
                f"is smaller than `max_seq_length` ({max_seq_length}). "
                "Using the tokenizer model max length instead."
            )
        assert max_seq_length is not None, (
            "If `block_size` is not set, `max_seq_length` must be set."
        )
        block_size = min(max_seq_length, tokenizer.model_max_length)
    else:
        if max_seq_length is not None:
            raise ValueError(
                "If `block_size` is set, `max_seq_length` should not be set."
                "The two parameters are aliases."
            )
        if tokenizer.model_max_length < block_size:
            logger.warning(
                f"Tokenizer model max length ({tokenizer.model_max_length}) "
                f"is smaller than `block_size` ({block_size}). "
                "Using the tokenizer model max length instead."
            )
        block_size = min(block_size, tokenizer.model_max_length)

    logger.info("Grouping data batches into block size (i.e., sequence size) "
                f"of {block_size}")

    def group_texts(ex):
        if not manually_add_eos:
            cat = {k: list(chain(*ex[k])) for k in ex}
            total = (len(cat["input_ids"]) // block_size) * block_size
            res = {
                k: [t[i:i + block_size]
                    for i in range(0, total, block_size)]
                for k, t in cat.items()
            }
            res["labels"] = res["input_ids"].copy()
            return res

        # manual eos insertion between documents before chunking
        eos_id = tokenizer.eos_token_id
        ids_cat, attn_cat = [], []
        for ids, attn in zip(ex["input_ids"], ex["attention_mask"]):
            ids_cat.extend(ids)
            attn_cat.extend(attn)
            if eos_id is not None:
                ids_cat.append(eos_id)
                attn_cat.append(1)

        total = (len(ids_cat) // block_size) * block_size
        input_blocks = [
            ids_cat[i:i + block_size] for i in range(0, total, block_size)
        ]
        attn_blocks = [
            attn_cat[i:i + block_size] for i in range(0, total, block_size)
        ]
        return {
            "input_ids": input_blocks,
            "attention_mask": attn_blocks,
            "labels": [b[:] for b in input_blocks],
        }

    with main_process_first("group texts"):
        if not streaming:
            lm_ds = tok_ds.map(
                group_texts,
                batched=True,
                num_proc=preprocessing_num_workers,
                load_from_cache_file=not overwrite_cache,
            )
        else:
            lm_ds = tok_ds.map(
                group_texts,
                batched=True,
            )

    train_ds = None
    eval_ds = None
    if do_train and "train" in lm_ds:
        train_ds = lm_ds["train"]
        if max_train_samples is not None:
            sz = min(len(train_ds), max_train_samples)
            train_ds = train_ds.select(range(sz))
    if do_eval and "validation" in lm_ds:
        eval_ds = lm_ds["validation"]
        if max_eval_samples is not None:
            sz = min(len(eval_ds), max_eval_samples)
            eval_ds = eval_ds.select(range(sz))

    if do_train:
        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)
    else:
        preprocess_logits_for_metrics = None
        compute_metrics = None

    if skip_samples is not None:
        assert skip_samples >= 0, "skip_samples must be non-negative"
        train_ds = train_ds.skip(skip_samples) if train_ds else None

    if mask_past_sequences:
        if ("flash_attention" not in attention_implementation) and (
                "flash_attn" not in attention_implementation):
            collator = BlockDiagFromEOSCollator(tokenizer)
        else:
            collator = BlockDiagFromEOSFA2Collator(
                tokenizer, ignore_idx=-100, return_seq_idx=False,)
    else:
        collator = default_data_collator

    return dict(
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
