import argparse
import glob
import json
import logging
import os
import random
import copy
import csv
import pandas as pd

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

#from apex.optimizers import FusedLAMB, FusedAdam
from apex.optimizers import FusedAdam
from schedulers import PolyWarmUpScheduler

from transformers import ( AdamW, get_linear_schedule_with_warmup ) 

from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import AlbertConfig, AlbertTokenizer, AlbertForSequenceClassification

##S-ERP 전처리기
import SerpPreProcessing

class GradientClipper:
    """
    Clips gradient norm of an iterable of parameters.
    """
    def __init__(self, max_grad_norm):
        self.max_norm = max_grad_norm
        if multi_tensor_applier.available:
            import amp_C
            self._overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
            self.multi_tensor_scale = amp_C.multi_tensor_scale
        else:
            raise RuntimeError('Gradient clipping requires cuda extensions')

    def step(self, params):
        l = [p.grad for p in params if p.grad is not None]
        self._overflow_buf.zero_()
        total_norm, _ = multi_tensor_applier(self.multi_tensor_l2norm, self._overflow_buf, [l], False)
        total_norm = total_norm.item()

        if (total_norm == float('inf')): return

        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            multi_tensor_applier(self.multi_tensor_scale, self._overflow_buf, [l, l], clip_coef)
        return True

# from kobert.pytorch_kobert import get_pytorch_kobert_model
# from kobert.utils import get_tokenizer

# from transformers import BertForSequenceClassification

logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False

class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are.
        This method converts examples to the correct format."""
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    @classmethod
    def _read_txt(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            lines = f.readlines()[1:]
            return lines

def is_sklearn_available():
    return _has_sklearn

def simple_accuracy(preds, labels):
	return (preds == labels).mean()

def acc_and_f1(preds, labels):
	acc = simple_accuracy(preds, labels)
	f1 = f1_score(y_true=labels, y_pred=preds)
	return {
		"acc": acc,
		"f1": f1,
		"acc_and_f1": (acc + f1) / 2,
}

def pearson_and_spearman(preds, labels):
	pearson_corr = pearsonr(preds, labels)[0]
	spearman_corr = spearmanr(preds, labels)[0]
	return {
		"pearson": pearson_corr,
		"spearmanr": spearman_corr,
		"corr": (pearson_corr + spearman_corr) / 2,
}

class SerpProcessor(DataProcessor):
    """Processor for the serp data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """Train Set data 넣어주기"""
        return self._create_examples(open(os.path.join(data_dir, "train.txt"), 'r', encoding="utf-8-sig").readlines(), "train")

    def get_dev_examples(self, data_dir):
        """Test Set data 넣어주기"""
        return self._create_examples(open(os.path.join(data_dir, "test.txt"), 'r', encoding="utf-8-sig").readlines(), "dev")
    
    def get_labels(self):
        """See base class."""
        list_f = open("label.txt", "r", encoding="utf-8-sig")                                
        labels = []
        for l in list_f.readlines():
            ##S-ERP - Start : 라벨 컨버전 / 삭제
            label = l[:-1]
            #label = pre.get_converted_label(label)
            if label not in labels:
                labels.append(label)
            ##S-ERP - End : 라벨 컨버전 / 삭제
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""    
        examples = []
        line_splited = False
        
        ##S-ERP - Start
        #### 샘플링 시작 ####        
        if set_type == 'train' and pre.args.pre_sample_train_min != -1:        
            lines = pre.sampling_train(lines)
            line_splited = True
        elif set_type == 'dev' and pre.args.pre_sample_test != -1:        
            lines = pre.sampling_test(lines)
            line_splited = True
        #### 샘플링 종료 ####
    
        i = 0 # True일 때만 i count    
        for line in lines: #enumerate(lines): 
            if line_splited == False:
                line = line[:-1].split('\t') 
            
            #### 전처리 수행 ####
            valid, label, text_a = pre.pre_process_module_cls(line)
            #print(valid, line[1], label)
            
            if valid == True:
                guid = "%s-%s" % (set_type, i) # True일 때만 i count
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
                i = i + 1
                #print(line[1], label)     
               
                # test면 상세 결과 출력용으로 보관
                if set_type == 'dev':
                    pre.record_test_data(line,text_a)       
        
            #### 전처리 종료 ####
        
        print("last of _create_examples for", set_type, " : ", examples[-1])
        ##S-ERP - End
        return examples

processors = {
    "serp" : SerpProcessor
}

output_modes = {
    "serp" : "classification"
}

def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}
    
    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length,)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":    
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )

    return features

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "serp":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

#ALL_MODELS = sum(
#    (
#        tuple(conf.pretrained_config_archive_map.keys())
#        for conf in (
#            AlbertConfig,
#        )
#    ),
#    (),
#)


MODEL_CLASSES = {
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):    
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu) // args.gradient_accumulation_steps
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #optimizer = FusedLAMB(optimizer_grouped_parameters, lr=args.learning_rate)

    # scheduler = get_linear_schedule_with_warmup(
    #    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    # )
    
    print('warmup steps : ', int(t_total * args.warmup_portions))
    scheduler = PolyWarmUpScheduler(optimizer, warmup=int(t_total * args.warmup_portions), total_steps=t_total, degree=1.0)


    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility

    # gradClipper = GradientClipper(max_grad_norm=1.0)    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            #if args.fp16:
            #    with amp.scale_loss(loss, optimizer) as scaled_loss:
            #        scaled_loss.backward()
            #else:
            #    loss.backward()

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(epoch_iterator) <= args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
            ):
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                # scheduler.step()
                model.zero_grad()
                global_step += 1
                
                #if args.fp16:
                #    gradClipper.step(amp.master_params(optimizer))
                #else:
                    # gradClipper.step(model.parameters())
                #    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # optimizer.step()
                #for param in model.parameters():
                #    param.grad = None
                # global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        ( args.local_rank == -1 or args.force_gpu_no != -1 ) and args.evaluate_during_training ## S-ERP                        
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                    ##S-ERP
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        results,softmax,f1_wgt,f1_mac = evaluate(args, model, tokenizer, "checkpoint-{}".format(global_step))
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value
                        

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    ##S-ERP - Start
    results = {}
    preds_lists = []
    preds_lists2 = []
    out_label_ids_lists = []
    logits_lists = []
    logits_lists2 = []
    ##S-ERP - End

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        modes = ["dev"]
        for mode in modes:
            eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, mode)
            
            if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(eval_output_dir)

            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu) // args.gradient_accumulation_steps
            # Note that DistributedSampler samples randomly
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

            # multi-gpu eval
            if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model)

            # Eval!
            logger.info("***** Running evaluation {} *****".format(prefix))
            logger.info("  Num examples = %d", len(eval_dataset))
            logger.info("  Batch size = %d", args.eval_batch_size)
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None            
            out_label_ids = None                        
            
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    if args.model_type != "distilbert":
                        inputs["token_type_ids"] = (
                            batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]                                       

                    eval_loss += tmp_eval_loss.mean().item()                    
                nb_eval_steps += 1

                ##S-ERP - Start      
                # softmax
                for sm in torch.nn.functional.softmax(logits, dim=1).to("cpu").numpy():
                    logits_lists.append(sm[np.argmax(sm)])
                    # for second highest
                    sm[np.argmax(sm)] = 0
                    preds_lists2.append(np.argmax(sm))
                    logits_lists2.append(sm[np.argmax(sm)])
                ##S-ERP - End

                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                    
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            if args.output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            elif args.output_mode == "regression":
                preds = np.squeeze(preds)
            
            preds_lists.extend(preds)            
            out_label_ids_lists.extend(out_label_ids)            
            
            #print(" preds_lists = ",  preds_lists) # 예측
            #print("out_label_ids_lists = ", out_label_ids_lists)    # 정답                                   
            #print("logits_lists = ", logits_lists)    # logits
            
            result = compute_metrics(eval_task, preds, out_label_ids)
            results.update(result)

            output_eval_file = os.path.join(eval_output_dir, prefix, mode + "_eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
   
    ##S-ERP - Start
    ####### 상세 결과 파일로 출력
    
    # Label로 변환
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    out_label_ids_lists_label = []
    preds_lists_label = []
    preds_lists2_label = []    
    for i in range(len(out_label_ids_lists)):
        out_label_ids_lists_label.append(label_list[out_label_ids_lists[i]])
        preds_lists_label.append(label_list[preds_lists[i]])
        preds_lists2_label.append(label_list[preds_lists2[i]])
        
    # 정답 확인
    ok_lists = []
    ok_lists2 = []
    for i in range(len(out_label_ids_lists_label)):
        if out_label_ids_lists_label[i] == preds_lists_label[i]:
            ok_lists.append(1)        
        else:
            ok_lists.append(0)

        if out_label_ids_lists_label[i] == preds_lists2_label[i]:
            ok_lists2.append(1)
        else:
            ok_lists2.append(0)

    # Data별 상세 결과
    result_df = pd.DataFrame.from_records(pre.test_data)
    result_df.insert(loc=0,column='Act',value=out_label_ids_lists_label)
    result_df.insert(loc=1,column='Pred1',value=preds_lists_label)
    result_df.insert(loc=2,column='OK1',value=ok_lists)
    result_df.insert(loc=3,column='softmax1',value=logits_lists)
    result_df.insert(loc=4,column='Pred2',value=preds_lists2_label)
    result_df.insert(loc=5,column='OK2',value=ok_lists2)
    result_df.insert(loc=6,column='softmax2',value=logits_lists2)
    result_df.to_csv(os.path.join(eval_output_dir, "_detail_result_"+prefix+".txt"), sep='\t', index=False, encoding = 'utf-8')

    # Label별 집계
    label_ok = result_df.groupby('Act').sum()['OK1']
    label_cnt = result_df.groupby('Act').count()['OK1']
    label_sm = result_df.groupby('Act').mean()['softmax1']
    label_result = pd.merge(label_cnt, label_ok, on='Act')    
    label_result = pd.merge(label_result, label_sm, on='Act')    
    label_result.columns = ['Count','OK', 'softmax']
    label_result['acc'] = label_result.apply(lambda x : x['OK'] / x['Count'] , axis=1)
    label_result['f1'] = label_result.apply(lambda x : f1_score(y_true=out_label_ids_lists_label, y_pred=preds_lists_label, labels=[x.name], average='macro'), axis=1)
    label_result.to_csv(os.path.join(eval_output_dir, "_result_by_labels_"+prefix+".txt"), sep='\t', index=True, encoding = 'utf-8')

    # Act-Pred Matrix
    label_matrix = result_df.groupby(['Act','Pred1']).count()[0]
    label_matrix.to_csv(os.path.join(eval_output_dir, "_label_matrix_"+prefix+".txt"), sep='\t', index=True, header=False, encoding = 'utf-8')
    
    # softmax 평균 반환
    softmax_mean = np.mean(logits_lists)
    
    # f1-score
    f1_wgt = f1_score(y_true=out_label_ids_lists_label, y_pred=preds_lists_label, average='weighted') # 표본수 가중치
    f1_mac = f1_score(y_true=out_label_ids_lists_label, y_pred=preds_lists_label, average='macro') # Label별 단순 평균
    logger.info("  %s = %s", 'f1_weighted', str(f1_wgt))
    logger.info("  %s = %s", 'f1_macro', str(f1_mac))

    return results, softmax_mean, f1_wgt, f1_mac
    ##S-ERP - End     

  
def load_and_cache_examples(args, task, tokenizer, mode): # mode is train, dev(eval), valid
    if args.local_rank not in [-1, 0] and mode == "train":
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        
        if mode == "train":
            examples = (processor.get_train_examples(args.data_dir))
        elif mode == "dev":
            examples = (processor.get_dev_examples(args.data_dir))
        elif mode == "valid":
            examples = (processor.get_val_examples(args.data_dir))

        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and mode == "train":
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        #help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(""),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--vocab_file",
        default="",
        type=str,
        help="vocab file path",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=20, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=20, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_portions", default=0.0, type=float, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=2000, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=88, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    ##S-ERP - Start
    ### START OF S-ERP argument ###
    parser.add_argument("--pre_conv_label", action="store_true", help="Convert label")
    parser.add_argument("--pre_delete_html", action="store_true", help="Delete HTML Tag")
    parser.add_argument("--pre_to_lower", action="store_true", help="Convert to lower case")
    parser.add_argument("--pre_delete_spc_chr", action="store_true", help="Delete special character")    
    parser.add_argument("--pre_delete_stopword", action="store_true", help="Delete stopword")    
    parser.add_argument("--pre_conv_num_to_zero", action="store_true", help="Convert number to 0000")    
    parser.add_argument("--pre_sample_train_min", default=-1, type=int, help="Min Number of train sample per label")
    parser.add_argument("--pre_sample_train_max", default=-1, type=int, help="Max Number of train sample per label")
    parser.add_argument("--pre_sample_test", default=-1, type=int, help="Number of test sample per label")
    parser.add_argument("--pre_sample_by", default="LATEST", type=str, help="Sampling method (LATEST or TEXT_LENGTH)")    
    parser.add_argument("--pre_convert_word", action="store_true", help="Convert word to similar word")    
    parser.add_argument("--force_gpu_no", default=-1, help="Force GPU no whe use 1 GPU only")
    ### END OF S-ERP argument ###
    ##S-ERP - End

    args = parser.parse_args()
        
    # Setup CUDA, GPU & distributed training    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)        
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device    

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    ##S-ERP - Start
    # SERP 전처리기 생성(global)
    global pre
    pre = SerpPreProcessing.SerpPreProcessing(args)
    ##S-ERP - End
    
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)    
    
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task='glue',
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
        
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=False, cls_token='<cls>', sep_token='<sep>'
    )

#    model = model_class.from_pretrained(
#        args.model_name_or_path,
#        from_tf=bool(".ckpt" in args.model_name_or_path),
#        config=config,
#        cache_dir=args.cache_dir if args.cache_dir else None,
#    )

    model = model_class(config)
    if args.model_name_or_path[-3:] == '.pt':
        checkpoint = torch.load(args.model_name_or_path, map_location="cpu")
        
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        if args.model_name_or_path[-1] != '/':
            args.model_name_or_path += '/'

        checkpoint = torch.load(args.model_name_or_path + 'pytorch_model.bin', map_location="cpu")
        checkpoint = {k: v for k, v in checkpoint.items() if 'albert' in k}
        print("checkpoint =", checkpoint.keys())
        model.load_state_dict(checkpoint, strict=False)    
        
    if args.local_rank == 0 and not args.no_cuda:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, "train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=False, cls_token='<cls>', sep_token='<sep>')
        model.to(args.device)

    # Evaluation
    results = {}
    softmax = [] ##S-ERP
    f1_wgt_list = [] ##S-ERP
    f1_mac_list = [] ##S-ERP
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=False, cls_token='<cls>', sep_token='<sep>')        
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + 'pytorch_model.bin', recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            ##S-ERP - Start
            result,smax,f1_wgt,f1_mac = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
            softmax.append(smax)
            f1_wgt_list.append(f1_wgt)
            f1_mac_list.append(f1_mac)
            ##S-ERP - End

        ##S-ERP - Start
        # 체크포인트별 출력   
        # 체크포인트별 acc/softamx/f1출력   
        import pandas as pd
        result_df = pd.DataFrame.from_dict(results,orient='index', columns=['acc'])
        result_df.insert(loc=1,column='softmax',value=softmax)  
        result_df.insert(loc=2,column='f1_weighted',value=f1_wgt_list)  
        result_df.insert(loc=3,column='f1_macro',value=f1_mac_list)  
        result_df.sort_values(['acc'], ascending=[False], inplace=True)        
        print(result_df)
        result_df.to_csv(os.path.join(args.output_dir, "acc_by_checkpoint.txt"), sep='\t', index=True, encoding = 'utf-8')
        ##S-ERP - End

if __name__ == "__main__":
    main()
    