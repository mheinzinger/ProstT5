"""
George Bouras 2024

This script is based off https://github.com/agemagician/ProtTrans/blob/master/Fine-Tuning/PT5_LoRA_Finetuning_per_residue_class.ipynb

For offline HPC use, you need to specify path to accuracy.py - https://github.com/huggingface/evaluate/issues/456

It can be downloaded using 

git clone https://github.com/huggingface/evaluate.git

It won't matter if you have access to the internet, no need to specify -a

# using conda to install the env (CUDA 12)

mamba create -n prostt5_training python=3.9
conda activate prostt5_training

mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install pandas
pip install transformers
pip install datasets
pip install evaluate
pip install matplotlib
pip install biopython
pip install ipykernel
pip install accuracy
pip install loguru


python finetune_prostt5_lora_script.py --trainaafasta train_aa.faa --trainthreedifasta train_3di.faa \
      --validaafasta valid_aa.faa  --validthreedifasta valid_3di.faa \
       -o test_prostt5_lora -b 1 --finetune_name prostt5_finetuned_model -f

I had trouble running deepspeed, so it is off by default. Use --deepspeed_flag to turn it on.

"""


# import dependencies
import argparse
import copy
import os
import os.path
import random
import re
import shutil
import sys
import time
from argparse import RawTextHelpFormatter

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from Bio import SeqIO
from datasets import Dataset
from evaluate import load
from loguru import logger
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorForTokenClassification,
    T5EncoderModel,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.t5.modeling_t5 import T5Config, T5PreTrainedModel, T5Stack
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

# Set environment variables to run Deepspeed from a notebook
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9993"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"



def get_input():
    parser = argparse.ArgumentParser(
        description="finetune_lora_script.py",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--trainaafasta",
        action="store",
        help="AA FASTA training dataset",
    )
    parser.add_argument(
        "--trainthreedifasta",
        action="store",
        help="Colabfold 3Di FASTA training dataset",
    )
    parser.add_argument(
        "--validaafasta",
        action="store",
        help="AA FASTA test dataset",
    )
    parser.add_argument(
        "--validthreedifasta",
        action="store",
        help="Colabfold 3Di FASTA test dataset",
    )
    parser.add_argument("-o", "--outdir", action="store", help="output directory.")
    parser.add_argument("-s", "--seed", default=13, action="store", type=int, help="seed")
    parser.add_argument("-b", "--batchsize", default=1, action="store", type=int, help="batchsize for GPU")
    parser.add_argument(
        "-f", "--force", help="Overwrites the output directory.", action="store_true"
    )
    parser.add_argument(
        "-m",
        "--model_dir",
        help="Path to save ProstT5_fp16 model to. Will automatically download"
    )
    parser.add_argument(
        "-a",
        "--accuracy_path",
        help="Path to accuracy.py - needed for offline use",
        default="accuracy"
    )
    parser.add_argument(
        "--deepspeed_flag", help="Turn on deepspeed.", action="store_true"
    )
    parser.add_argument(
        "--finetune_name",
        default="prostt5_finetuned_model",
        help="Name of finetuned model."
    )



    args = parser.parse_args()

    return args


def main():
    # get the args
    args = get_input()

    logger.add(lambda _: sys.exit(1), level="ERROR")

    if args.force == True:
        if os.path.isdir(args.outdir) == True:
            print(
                f"Removing output directory {args.outdir} as -f or --force was specified."
            )
            shutil.rmtree(args.outdir)
        elif os.path.isfile(args.outdir) == True:
            os.remove(args.outdir)
        else:
            print(
                f"--force was specified even though the output directory {args.outdir} does not already exist. Continuing."
            )
    else:
        if os.path.isdir(args.outdir) == True or os.path.isfile(args.outdir) == True:
            print(
                f"The output directory {args.outdir} already exists and force was not specified. Please specify -f or --force to overwrite it."
            )
            sys.exit()

    # make dir
    if os.path.isdir(args.outdir) == False:
        os.mkdir(args.outdir)

    # initial logging stuff
    start_time = time.time()
    log_file = os.path.join(args.outdir, f"Prostt5_LoRA_{start_time}.log")
    # adds log file
    logger.add(log_file)

    logger.info(f"Torch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}" )
    logger.info(f"Cuda version:  {torch.version.cuda}")
    logger.info(f"Numpy version:  {np.__version__}")
    logger.info(f"Pandas version:  {pd.__version__}")
    logger.info(f"Transformers version:  {transformers.__version__}")
    logger.info(f"Datasets version: {datasets.__version__}")

    logger.info("Creating dataset")


    logger.add(lambda _: sys.exit(1), level="ERROR")

    if args.force == True:
        if os.path.isdir(args.outdir) == True:
            print(
                f"Removing output directory {args.outdir} as -f or --force was specified."
            )
            shutil.rmtree(args.outdir)
        elif os.path.isfile(args.outdir) == True:
            os.remove(args.outdir)
        else:
            print(
                f"--force was specified even though the output directory {args.outdir} does not already exist. Continuing."
            )
    else:
        if os.path.isdir(args.outdir) == True or os.path.isfile(args.outdir) == True:
            print(
                f"The output directory {args.outdir} already exists and force was not specified. Please specify -f or --force to overwrite it."
            )
            sys.exit()

    # make dir
    if os.path.isdir(args.outdir) == False:
        os.mkdir(args.outdir)

    # initial logging stuff
    start_time = time.time()
    log_file = os.path.join(args.outdir, f"Prostt5_LoRA_{start_time}.log")
    # adds log file
    logger.add(log_file)

    logger.info(f"Torch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}" )
    logger.info(f"Cuda version:  {torch.version.cuda}")
    logger.info(f"Numpy version:  {np.__version__}")
    logger.info(f"Pandas version:  {pd.__version__}")
    logger.info(f"Transformers version:  {transformers.__version__}")
    logger.info(f"Datasets version: {datasets.__version__}")

    logger.info("Creating dataset")

    # AA Sequence File
    logger.info("Reading AA Fastas")

    # Load the AA fasta files
    fasta_path = args.trainaafasta

    # mask 0s if not in set
    standard_aa_set = set("ACDEFGHIKLMNPQRSTVWY")
    with open(fasta_path, "r") as fasta_file:
        sequences = [
            [record.id, str(record.seq), [0 if char not in standard_aa_set else 1 for char in str(record.seq)]]
            for record in SeqIO.parse(fasta_file, "fasta")
        ]

    # Create dataframe
    df_train_aa = pd.DataFrame(sequences, columns=["name", "sequence", "mask"])
    df_train_aa["set"] = 'train'


    fasta_path = args.validaafasta

    # mask 0s if not a standard AA
    standard_aa_set = set("ACDEFGHIKLMNPQRSTVWY")
    with open(fasta_path, "r") as fasta_file:
        sequences = [
            [record.id, str(record.seq), [0 if char not in standard_aa_set else 1 for char in str(record.seq)]]
            for record in SeqIO.parse(fasta_file, "fasta")
        ]

    df_valid_aa = pd.DataFrame(sequences, columns=["name", "sequence", "mask"])
    df_valid_aa["set"] = 'valid'

    # combine
    df_aa = pd.concat([df_train_aa, df_valid_aa], ignore_index=True)
    df_aa.to_csv(f"{args.outdir}/aa_fasta.tsv", index=False, sep="\t")

    #####
    # 3Di fasta

    # AA Sequence File
    logger.info("Reading Colabfold 3Di Fastas")

    ss_mapping = {
        0: "A",
        1: "C",
        2: "D",
        3: "E",
        4: "F",
        5: "G",
        6: "H",
        7: "I",
        8: "K",
        9: "L",
        10: "M",
        11: "N",
        12: "P",
        13: "Q",
        14: "R",
        15: "S",
        16: "T",
        17: "V",
        18: "W",
        19: "Y",
    }

    # train
    threedi_fasta_path = args.trainthreedifasta

    sequences = []
    with open(threedi_fasta_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequence_numbers = [
                next(key for key, value in ss_mapping.items() if value == aa)
                for aa in str(record.seq)
            ]
            sequences.append([str(record.id), sequence_numbers])

    # Create dataframe
    df_train_3di = pd.DataFrame(sequences, columns=["name", "label"])

    # valid
    threedi_fasta_path = args.validthreedifasta

    sequences = []
    with open(threedi_fasta_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequence_numbers = [
                next(key for key, value in ss_mapping.items() if value == aa)
                for aa in str(record.seq)
            ]
            sequences.append([str(record.id), sequence_numbers])

    # Create dataframe
    df_valid_3di = pd.DataFrame(sequences, columns=["name", "label"])

    # combine
    df_3di = pd.concat([df_train_3di, df_valid_3di], ignore_index=True)
    df_3di.to_csv(f"{args.outdir}/3di_fasta.tsv", index=False, sep="\t")


    logger.info("Merging AA and 3Di into one dataframe")
    merged_df = pd.merge(df_aa, df_3di, on="name", how="left")

    # gets the train and valid dataset
    logger.info("Getting train and validation datasets")
    my_train = merged_df[merged_df['set'] == 'train']
    my_valid = merged_df[merged_df['set'] == 'valid']
       
    
    # Set labels where mask == 0 to -100 (will be ignored by pytorch loss)
    my_train['label'] = my_train.apply(lambda row: [-100 if m == 0 else l for l, m in zip(row['label'], row['mask'])], axis=1)
    my_valid['label'] = my_valid.apply(lambda row: [-100 if m == 0 else l for l, m in zip(row['label'], row['mask'])], axis=1)

    # save the dataset
    my_train.to_pickle(f"{args.outdir}/train.pkl")
    my_valid.to_pickle(f"{args.outdir}/valid.pkl")
    my_train.to_csv(f"{args.outdir}/train.tsv", index=False, sep="\t")
    my_valid.to_csv(f"{args.outdir}/valid.tsv", index=False, sep="\t")



    # Modifies an existing transformer and introduce the LoRA layers

    class LoRAConfig:
        def __init__(self):
            self.lora_rank = 4
            self.lora_init_scale = 0.01
            self.lora_modules = ".*SelfAttention|.*EncDecAttention"
            self.lora_layers = "q|k|v|o"
            self.trainable_param_names = ".*layer_norm.*|.*lora_[ab].*"
            self.lora_scaling_rank = 1
            # lora_modules and lora_layers are speicified with regular expressions
            # see https://www.w3schools.com/python/python_regex.asp for reference

    class LoRALinear(nn.Module):
        def __init__(self, linear_layer, rank, scaling_rank, init_scale):
            super().__init__()
            self.in_features = linear_layer.in_features
            self.out_features = linear_layer.out_features
            self.rank = rank
            self.scaling_rank = scaling_rank
            self.weight = linear_layer.weight
            self.bias = linear_layer.bias
            if self.rank > 0:
                self.lora_a = nn.Parameter(
                    torch.randn(rank, linear_layer.in_features) * init_scale
                )
                if init_scale < 0:
                    self.lora_b = nn.Parameter(
                        torch.randn(linear_layer.out_features, rank) * init_scale
                    )
                else:
                    self.lora_b = nn.Parameter(
                        torch.zeros(linear_layer.out_features, rank)
                    )
            if self.scaling_rank:
                self.multi_lora_a = nn.Parameter(
                    torch.ones(self.scaling_rank, linear_layer.in_features)
                    + torch.randn(self.scaling_rank, linear_layer.in_features)
                    * init_scale
                )
                if init_scale < 0:
                    self.multi_lora_b = nn.Parameter(
                        torch.ones(linear_layer.out_features, self.scaling_rank)
                        + torch.randn(linear_layer.out_features, self.scaling_rank)
                        * init_scale
                    )
                else:
                    self.multi_lora_b = nn.Parameter(
                        torch.ones(linear_layer.out_features, self.scaling_rank)
                    )

        def forward(self, input):
            if self.scaling_rank == 1 and self.rank == 0:
                # parsimonious implementation for ia3 and lora scaling
                if self.multi_lora_a.requires_grad:
                    hidden = F.linear(
                        (input * self.multi_lora_a.flatten()), self.weight, self.bias
                    )
                else:
                    hidden = F.linear(input, self.weight, self.bias)
                if self.multi_lora_b.requires_grad:
                    hidden = hidden * self.multi_lora_b.flatten()
                return hidden
            else:
                # general implementation for lora (adding and scaling)
                weight = self.weight
                if self.scaling_rank:
                    weight = (
                        weight
                        * torch.matmul(self.multi_lora_b, self.multi_lora_a)
                        / self.scaling_rank
                    )
                if self.rank:
                    weight = weight + torch.matmul(self.lora_b, self.lora_a) / self.rank

                # Convert weight to half (float16) precision to match unput
                # weight = weight.half()
                # input = input.half()

                # convert both to float
                input = input.to(torch.half)
                weight = weight.to(torch.half)

                return F.linear(input, weight, self.bias)

        def extra_repr(self):
            return "in_features={}, out_features={}, bias={}, rank={}, scaling_rank={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.rank,
                self.scaling_rank,
            )

    def modify_with_lora(transformer, config):
        for m_name, module in dict(transformer.named_modules()).items():
            if re.fullmatch(config.lora_modules, m_name):
                for c_name, layer in dict(module.named_children()).items():
                    if re.fullmatch(config.lora_layers, c_name):
                        assert isinstance(
                            layer, nn.Linear
                        ), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                        setattr(
                            module,
                            c_name,
                            LoRALinear(
                                layer,
                                config.lora_rank,
                                config.lora_scaling_rank,
                                config.lora_init_scale,
                            ),
                        )
        return transformer

    class ClassConfig:
        def __init__(self, dropout=0.2, num_labels=3):
            self.dropout_rate = dropout
            self.num_labels = num_labels

    class T5EncoderForTokenClassification(T5PreTrainedModel):
        def __init__(self, config: T5Config, class_config):
            super().__init__(config)
            self.num_labels = class_config.num_labels
            self.config = config

            self.shared = nn.Embedding(config.vocab_size, config.d_model)

            encoder_config = copy.deepcopy(config)
            encoder_config.use_cache = False
            encoder_config.is_encoder_decoder = False
            self.encoder = T5Stack(encoder_config, self.shared)

            self.dropout = nn.Dropout(class_config.dropout_rate)
            self.classifier = nn.Linear(config.hidden_size, class_config.num_labels)

            # Initialize weights and apply final processing
            self.post_init()

            # Model parallel
            self.model_parallel = False
            self.device_map = None

        def parallelize(self, device_map=None):
            self.device_map = (
                get_device_map(
                    len(self.encoder.block), range(torch.cuda.device_count())
                )
                if device_map is None
                else device_map
            )
            assert_device_map(self.device_map, len(self.encoder.block))
            self.encoder.parallelize(self.device_map)
            self.classifier = self.classifier.to(self.encoder.first_device)
            self.model_parallel = True

        def deparallelize(self):
            self.encoder.deparallelize()
            self.encoder = self.encoder.to("cpu")
            self.model_parallel = False
            self.device_map = None
            torch.cuda.empty_cache()

        def get_input_embeddings(self):
            return self.shared

        def set_input_embeddings(self, new_embeddings):
            self.shared = new_embeddings
            self.encoder.set_input_embeddings(new_embeddings)

        def get_encoder(self):
            return self.encoder

        def _prune_heads(self, heads_to_prune):
            """
            Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
            class PreTrainedModel
            """
            for layer, heads in heads_to_prune.items():
                self.encoder.layer[layer].attention.prune_heads(heads)

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )

            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = outputs[0]
            sequence_output = self.dropout(sequence_output)
            # Convert the tensor_half to torch.float32 before the operation
            sequence_output_fl = sequence_output.to(torch.float32)
            logits = self.classifier(sequence_output_fl)

            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()

                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)

                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(-100).type_as(labels)
                )

                valid_logits = active_logits[active_labels != -100]
                valid_labels = active_labels[active_labels != -100]

                valid_labels = valid_labels.type(torch.LongTensor).to("cuda:0")

                loss = loss_fct(valid_logits, valid_labels)

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    def PT5_classification_model(num_labels, model_dir, half_precision=False):
        # Load PT5 and tokenizer
        # possible to load the half preciion model (thanks to @pawel-rezo for pointing that out)
        if not half_precision:
            model = T5EncoderModel.from_pretrained(
                f"Rostlab/ProstT5", cache_dir=f"{model_dir}/"
            )
            tokenizer = T5Tokenizer.from_pretrained(
                f"Rostlab/ProstT5", cache_dir=f"{model_dir}/"
            )
        elif half_precision and torch.cuda.is_available():
            tokenizer = T5Tokenizer.from_pretrained(
                f"Rostlab/ProstT5_fp16", do_lower_case=False, cache_dir=f"{model_dir}/"
            )
            model = T5EncoderModel.from_pretrained(
                f"Rostlab/ProstT5_fp16",
                torch_dtype=torch.float16,
                cache_dir=f"{model_dir}/",
            ).to(torch.device("cuda"))
        else:
            raise ValueError("Half precision can be run on GPU only.")

        # Create new Classifier model with PT5 dimensions
        class_config = ClassConfig(num_labels=num_labels)
        class_model = T5EncoderForTokenClassification(model.config, class_config)

        # Set encoder and embedding weights to checkpoint weights
        class_model.shared = model.shared
        class_model.encoder = model.encoder

        # Delete the checkpoint model
        model = class_model
        del class_model

        # Print number of trainable parameters
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("ProstT5_Classfier\nTrainable Parameter: " + str(params))

        # Add model modification lora
        config = LoRAConfig()

        # Add LoRA layers
        model = modify_with_lora(model, config)

        # Freeze Embeddings and Encoder (except LoRA)
        for param_name, param in model.shared.named_parameters():
            param.requires_grad = False
        for param_name, param in model.encoder.named_parameters():
            param.requires_grad = False

        for param_name, param in model.named_parameters():
            if re.fullmatch(config.trainable_param_names, param_name):
                param.requires_grad = True

        # Print trainable Parameter
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("ProstT5_LoRA_Classfier\nTrainable Parameter: " + str(params) + "\n")

        return model, tokenizer

    # Deepspeed config for optimizer CPU offload

    ds_config = {
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
            },
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 2000,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }

    # Set random seeds for reproducibility of your trainings run
    def set_seeds(s):
        torch.manual_seed(s)
        np.random.seed(s)
        random.seed(s)
        set_seed(s)

    # Dataset creation
    def create_dataset(tokenizer, seqs, labels):
        tokenized = tokenizer(seqs, max_length=1024, padding=False, truncation=True)
        dataset = Dataset.from_dict(tokenized)
        # we need to cut of labels after 1023 positions for the data collator to add the correct padding (1023 + 1 special tokens)
        labels = [l[:1023] for l in labels]
        dataset = dataset.add_column("labels", labels)

        return dataset

    # Main training fuction
    def train_per_residue(
        train_df,  # training data
        valid_df,  # validation data
        num_labels=3,  # number of classes
        # effective training batch size is batch * accum
        # we recommend an effective batch size of 8
        batch=4,  # for training
        accum=2,  # gradient accumulation
        val_batch=16,  # batch size for evaluation
        epochs=10,  # training epochs
        lr=3e-4,  # recommended learning rate
        seed=42,  # random seed
        deepspeed=True,  # if gpu is large enough disable deepspeed for training speedup
        mixed=False,  # enable mixed precision training
        gpu=1,
        model_dir='model',
        accuracy_path = 'accuracy'
    ):  # gpu selection (1 for first gpu)
        # Set gpu device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu - 1)

        # Set all random seeds
        set_seeds(int(seed))

        # load model
        model, tokenizer = PT5_classification_model(
            num_labels=num_labels, model_dir=model_dir
        )

        # Preprocess inputs
        # Replace uncommon AAs with "X"
        train_df["sequence"] = train_df["sequence"].str.replace(
            "|".join(["O", "B", "U", "Z"]), "X", regex=True
        )
        valid_df["sequence"] = valid_df["sequence"].str.replace(
            "|".join(["O", "B", "U", "Z"]), "X", regex=True
        )
        # Add spaces between each amino acid for PT5 to correctly use them
        train_df["sequence"] = train_df.apply(
            lambda row: " ".join(row["sequence"]), axis=1
        )
        valid_df["sequence"] = valid_df.apply(
            lambda row: " ".join(row["sequence"]), axis=1
        )

        # Create Datasets
        train_set = create_dataset(
            tokenizer, list(train_df["sequence"]), list(train_df["label"])
        )
        valid_set = create_dataset(
            tokenizer, list(valid_df["sequence"]), list(valid_df["label"])
        )

        # Huggingface Trainer arguments
        args = TrainingArguments(
            "./",
            evaluation_strategy="steps",
            eval_steps=500,
            logging_strategy="epoch",
            save_strategy="no",
            learning_rate=lr,
            per_device_train_batch_size=batch,
            # per_device_eval_batch_size=val_batch,
            per_device_eval_batch_size=batch,
            gradient_accumulation_steps=accum,
            num_train_epochs=epochs,
            seed=seed,
            deepspeed=ds_config if deepspeed else None,
            fp16=mixed,
        )

        # Metric definition for validation data
        def compute_metrics(eval_pred):
            # path to the accuracy.py file
            metric = load(accuracy_path)
            predictions, labels = eval_pred

            labels = labels.reshape((-1,))

            predictions = np.argmax(predictions, axis=2)
            predictions = predictions.reshape((-1,))

            predictions = predictions[labels != -100]
            labels = labels[labels != -100]

            return metric.compute(predictions=predictions, references=labels)

        # For token classification we need a data collator here to pad correctly
        data_collator = DataCollatorForTokenClassification(tokenizer)

        # Trainer
        trainer = Trainer(
            model,
            args,
            train_dataset=train_set,
            eval_dataset=valid_set,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Train model
        trainer.train()

        return tokenizer, model, trainer.state.log_history

    torch.cuda.empty_cache()

    logger.info("Finetuning Prostt5")

    # 20 labels for ProstT5
    label_count = 20
    learning_rate = 1e-4
    epoch_count = 1
    batch_size = args.batchsize
    accum_rate = 2



    tokenizer, model, history = train_per_residue(
        my_train,
        my_valid,
        num_labels=label_count,
        batch=batch_size,
        lr=learning_rate,
        accum=accum_rate,
        epochs=epoch_count,
        seed=13,
        gpu=1,
        deepspeed=args.deepspeed_flag,
        model_dir=args.model_dir,
        accuracy_path=args.accuracy_path,
        mixed = True
    )

    # Get loss, val_loss, and the computed metric from history
    loss = [x["loss"] for x in history if "loss" in x]
    val_loss = [x["eval_loss"] for x in history if "eval_loss" in x]

    # Get accuracy value
    metric = [x["eval_accuracy"] for x in history if "eval_accuracy" in x]

    ## plot

    # Get loss, val_loss, and the computed metric from history
    loss = [x["loss"] for x in history if "loss" in x]
    val_loss = [x["eval_loss"] for x in history if "eval_loss" in x]

    # Get accuracy value
    metric = [x["eval_accuracy"] for x in history if "eval_accuracy" in x]

    epochs_loss = [x["epoch"] for x in history if "loss" in x]
    epochs_eval = [x["epoch"] for x in history if "eval_loss" in x]

    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    # Plot loss and val_loss on the first y-axis
    # For the loss we plot a horizontal line because we have just one loss value (after the first epoch)
    # Exchange the two lines below if you trained multiple epochs
    line1 = ax1.plot([0] + epochs_loss, loss * 2, label="train_loss")
    # line1 = ax1.plot(epochs_loss, loss, label='train_loss')

    line2 = ax1.plot(epochs_eval, val_loss, label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    # Plot the computed metric on the second y-axis
    line3 = ax2.plot(epochs_eval, metric, color="red", label="val_accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0, 1])

    # Combine the lines from both y-axes and create a single legend
    lines = line1 + line2 + line3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower left")

    # Show the plot
    plt.title("Training History")
    plt.savefig(f"{args.outdir}/loss_plot.png")
    plt.show()

    def save_model(model, filepath):
        # Saves all parameters that were changed during finetuning

        # Create a dictionary to hold the non-frozen parameters
        non_frozen_params = {}

        # Iterate through all the model parameters
        for param_name, param in model.named_parameters():
            # If the parameter has requires_grad=True, add it to the dictionary
            if param.requires_grad:
                non_frozen_params[param_name] = param

        # Save only the finetuned parameters
        torch.save(non_frozen_params, filepath)

    # def load_model(filepath, num_labels=1, mixed=True):
    #     # Creates a new PT5 model and loads the finetuned weights from a file

    #     # load a new model
    #     model, tokenizer = PT5_classification_model(
    #         num_labels=num_labels, half_precision=mixed
    #     )

    #     # Load the non-frozen parameters from the saved file
    #     non_frozen_params = torch.load(filepath)

    #     # Assign the non-frozen parameters to the corresponding parameters of the model
    #     for param_name, param in model.named_parameters():
    #         if param_name in non_frozen_params:
    #             param.data = non_frozen_params[param_name].data

    #     return tokenizer, model

    save_model(model, f"{args.outdir}/{args.finetune_name}.pth")

    logger.info("Finetuning complete :)")


if __name__ == "__main__":
    main()
