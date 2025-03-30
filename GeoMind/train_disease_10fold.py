import evaluate
import os 
# from huggingface_hub import login
from huggingface_hub import login
from sklearn.model_selection import KFold
# login(token="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")
from dataset2 import Imdb_dataset, Dataset_ADNI_seqFC, Dataset_OASIS_seqFC,Dataset_PPMI_seqFC,SeqFCDataset_7tasks,Dataset_WM, Dataset_HCP_WM, SeqFCDataset_WM, Dataset_ADNI_seq, Dataset_OASIS_seq, Dataset_PPMI_seq, Dataset_HCPA_seq
from datasets import load_dataset
from utils2 import compute_metrics
from transformers import AutoTokenizer , TrainingArguments
from model2 import MambaTextClassification, MambaTrainer, MambaConfig
from torch.utils.data import DataLoader, ConcatDataset, random_split
from sklearn.model_selection import KFold
import torch
import numpy as np
import random
import warnings
import argparse
warnings.filterwarnings("ignore")
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ADNI', required=False, help="Choose the dataset")
parser.add_argument('--gpu', type=str,default='cuda:1', required=False, help="Choose the gpu")
parser.add_argument('--batch', type=int,default=16, required=False, help="Choose the batch size")
parser.add_argument('--dim', type=int,default=116, required=False, help="Choose the dim")
parser.add_argument('--state', type=int,default=2, required=False, help="Choose the number of clusters")
parser.add_argument('--epoch', type=int,default=300, required=False, help="Choose the number of epochs")
parser.add_argument('--fold', type=int,default=10, required=False, help="Choose the number of folds")
parser.add_argument('--hidden', type=int,default=2048, required=False, help="Choose the dimension of hidden state")
parser.add_argument('--layer', type=int,default=2, required=False, help="Choose the number of layers")

args = parser.parse_args()

if args.dataset == 'ADNI':
    input_dir = f'**'
    label_dir = '**'
    # dataset = Dataset_ADNI_seqFC(input_dir, label_dir) #geo_mamba
    dataset = Dataset_ADNI_seq(input_dir, label_dir)
    time_points = 140
elif args.dataset == 'HCPYA':
    input_dir = f'**'
    label_dir = '**'
    dataset = SeqFCDataset_WM(input_dir, label_dir)
    # dataset = Dataset_HCP_WM(input_dir, label_dir)
    time_points = 39
elif args.dataset == 'OASIS':
    input_dir = f'**'
    # label_dir = '/path/to/OASIS/label'
    # dataset = Dataset_OASIS_seqFC(input_dir)
    dataset = Dataset_OASIS_seq(input_dir)
    time_points = 328
elif args.dataset in ['PPMI', 'ABIDE', 'neurocon','taowu']  :
    input_dir = f'**/{args.dataset}'
    # label_dir = '/path/to/OASIS/label'
    # dataset = Dataset_PPMI_seqFC(input_dir)
    dataset = Dataset_PPMI_seq(input_dir)
    time_points = 137
else:
    raise ValueError(f"Dataset {args.dataset} not supported.")

# Load the dataset
# train_dataset = Dataset_HCP_WM(input_dir, label_dir)
# test_dataset = Dataset_HCP_WM(input_test, label_test)

kf = KFold(n_splits=args.fold, shuffle=False, random_state=None)

all_metrics = {
    "accuracy": [],
    "precision": [],
    "f1": [],
    "recall": []
}

config = MambaConfig(
    d_model=args.hidden,
    in_feas=args.dim,
    n_layer=args.layer,
    vocab_size=137,
    ssm_cfg={},
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    d_state=args.state
)

output_file = f'Results/CV_{args.dataset}_{args.state}_{args.fold}fold_{args.hidden}_{args.layer}.txt'
# Cross-validation
# Open the file for writing
with open(output_file, 'w') as f:

    # Cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        f.write(f"Training fold {fold + 1}...\n")
        print(f"Training fold {fold + 1}...")

        # Split dataset into training and validation sets for this fold
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # Re-initialize the model for each fold
        model = MambaTextClassification(config).to(torch.device(args.gpu))

        # Training arguments
        arguments = TrainingArguments(
            output_dir=f'mamba_text_classification_fold{fold + 1}',
            learning_rate=1e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            num_train_epochs=args.epoch,
            warmup_steps=0.01,
            lr_scheduler_type='cosine',
            evaluation_strategy='steps',
            eval_steps=20,
            load_best_model_at_end=True,
            report_to=None
        )

        # Trainer for the fold
        trainer = MambaTrainer(
            model=model,
            args=arguments,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            tokenizer=None,
            compute_metrics=compute_metrics
        )

        # Train the model
        trainer.train()

        # Evaluate the model and save the metrics
        metrics = trainer.evaluate(eval_dataset=val_subset)
        f.write(f"Fold {fold + 1} metrics: {metrics}\n")
        print(f"Fold {fold + 1} metrics: {metrics}")

        # Append metrics to the respective lists
        all_metrics["accuracy"].append(metrics["eval_accuracy"])
        all_metrics["precision"].append(metrics["eval_precision"])
        all_metrics["f1"].append(metrics["eval_f1"])
        all_metrics["recall"].append(metrics["eval_recall"])

    # Compute mean and variance of the metrics across all folds
    mean_metrics = {metric: np.mean(all_metrics[metric]) for metric in all_metrics}
    var_metrics = {metric: np.var(all_metrics[metric]) for metric in all_metrics}

    # Write the mean and variance results to the file
    f.write("\nMean metrics:\n")
    for key, value in mean_metrics.items():
        f.write(f"{key}: {value}\n")

    f.write("\nVariance of metrics:\n")
    for key, value in var_metrics.items():
        f.write(f"{key}: {value}\n")

print("Results saved to:", output_file)