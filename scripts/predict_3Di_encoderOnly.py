#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:27:44 2023

@author: mheinzinger
"""

import argparse
import time
from pathlib import Path

from urllib import request
import shutil

import numpy as np
import torch
from torch import nn
from transformers import T5EncoderModel, T5Tokenizer


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))


# Convolutional neural network (two convolutional layers)
class CNN(nn.Module):
    def __init__( self ):
        super(CNN, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Conv2d(32, 20, kernel_size=(7, 1), padding=(3, 0))
        )

    def forward(self, x):
        """
            L = protein length
            B = batch-size
            F = number of features (1024 for embeddings)
            N = number of classes (20 for 3Di)
        """
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)  # IN: X = (B x L x F); OUT: (B x F x L, 1)
        Yhat = self.classifier(x)  # OUT: Yhat_consurf = (B x N x L x 1)
        Yhat = Yhat.squeeze(dim=-1)  # IN: (B x N x L x 1); OUT: ( B x L x N )
        return Yhat


def get_T5_model(model_dir):
    print("Loading T5 from: {}".format(model_dir))
    model = T5EncoderModel.from_pretrained(model_dir).to(device)
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False )
    return model, vocab


def read_fasta( fasta_path, split_char, id_field ):
    '''
        Reads in fasta file containing multiple sequences.
        Returns dictionary of holding multiple sequences or only single 
        sequence, depending on input file.
    '''
    
    sequences = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                sequences[ uniprot_id ] = ''
            else:
                s = ''.join( line.split() ).replace("-","")
                
                if s.islower(): # sanity check to avoid mix-up of 3Di and AA input
                    print("The input file was in lower-case which indicates 3Di-input." + 
                          "This predictor only operates on amino-acid-input (upper-case)." +
                          "Exiting now ..."
                          )
                    return None
                else:
                    sequences[ uniprot_id ] += s
    return sequences

def write_predictions(predictions, out_path):
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
        19: "Y"
    }
    
    with open(out_path, 'w+') as out_f:
        out_f.write( '\n'.join(
          [ ">{}\n{}".format(
              seq_id, "".join(list(map(lambda yhat: ss_mapping[int(yhat)], yhats))) )
          for seq_id, yhats in predictions.items()
          ]
            ) )
    print(f"Finished writing results to {out_path}")
    return None


def toCPU(tensor):
    if len(tensor.shape) > 1:
        return tensor.detach().cpu().squeeze(dim=-1).numpy()
    else:
        return tensor.detach().cpu().numpy()


def download_file(url,local_path):
    if not local_path.parent.is_dir():
        local_path.parent.mkdir()
        
    print("Downloading: {}".format(url))
    req = request.Request(url, headers={
          'User-Agent' : 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
      })
  
    with request.urlopen(req) as response, open(local_path, 'wb') as outfile:
          shutil.copyfileobj(response, outfile)
    return None

    
def load_predictor( weights_link="https://rostlab.org/~deepppi/prostt5/cnn_chkpnt/model.pt" ):
    model = CNN()
    checkpoint_p = Path.cwd() / "cnn_chkpnt" / "model.pt"
    # if no pre-trained model is available, yet --> download it
    if not checkpoint_p.exists():
        download_file(weights_link, checkpoint_p)

    # Torch load will map back to device from state, which often is GPU:0.
    # to overcome, need to explicitly map to active device
    global device

    state = torch.load(checkpoint_p, map_location=device)

    model.load_state_dict(state["state_dict"])

    model = model.eval()
    model = model.to(device)

    return model


def get_embeddings( seq_path, out_path, model_dir, split_char, id_field, half_precision,    
                   max_residues=4000, max_seq_len=1000, max_batch=100 ):
    
    seq_dict = dict()
    predictions = dict()

    # Read in fasta
    seq_dict = read_fasta( seq_path, split_char, id_field )
    prefix = "<AA2fold>"
    
    model, vocab = get_T5_model(model_dir)
    predictor = load_predictor()
    
    if half_precision:
        model = model.half()
        predictor = predictor.half()
        print("Using models in half-precision.")
    else:
        model = model.full()
        predictor = predictor.full()
        print("Using models in full-precision.")
        
    print('########################################')
    print('Example sequence: {}\n{}'.format( next(iter(
            seq_dict.keys())), next(iter(seq_dict.values()))) )
    print('########################################')
    print('Total number of sequences: {}'.format(len(seq_dict)))

    avg_length = sum([ len(seq) for _, seq in seq_dict.items()]) / len(seq_dict)
    n_long     = sum([ 1 for _, seq in seq_dict.items() if len(seq)>max_seq_len])
    # sort sequences by length to trigger OOM at the beginning
    seq_dict   = sorted( seq_dict.items(), key=lambda kv: len( seq_dict[kv[0]] ), reverse=True )
    
    print("Average sequence length: {}".format(avg_length))
    print("Number of sequences >{}: {}".format(max_seq_len, n_long))
    
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict,1):
        # replace non-standard AAs
        seq = seq.replace('U','X').replace('Z','X').replace('O','X')
        seq_len = len(seq)
        seq = prefix + ' ' + ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed 
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len 
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            token_encoding = vocab.batch_encode_plus(seqs, 
                                                     add_special_tokens=True, 
                                                     padding="longest", 
                                                     return_tensors='pt' 
                                                     ).to(device)
            try:
                with torch.no_grad():
                    embedding_repr = model(token_encoding.input_ids, 
                                           attention_mask=token_encoding.attention_mask
                                           )
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(
                    pdb_id, seq_len)
                    )
                continue
            
            # ProtT5 appends a special tokens at the end of each sequence
            # Mask this also out during inference while taking into account the prefix
            for idx, s_len in enumerate(seq_lens):
                token_encoding.attention_mask[idx,s_len+1] = 0

            # extract last hidden states (=embeddings)
            residue_embedding = embedding_repr.last_hidden_state.detach()
            # mask out padded elements in the attention output (can be non-zero) for further processing/prediction
            residue_embedding = residue_embedding*token_encoding.attention_mask.unsqueeze(dim=-1)
            # slice off embedding of special token prepended before to each sequence
            residue_embedding = residue_embedding[:,1:]
            
            prediction = predictor(residue_embedding)
            prediction = toCPU(torch.max( prediction, dim=1, keepdim=True )[1] ).astype(np.byte)

            # batch-size x seq_len x embedding_dim
            # extra token is added at the end of the seq
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                # slice off padding and special token appended to the end of the sequence
                predictions[identifier] = prediction[batch_idx,:, 0:s_len].squeeze()
                assert s_len == len(predictions[identifier]), print(f"Length mismatch for {identifier}: is:{len(predictions[identifier])} vs should:{s_len}")
                if len(predictions) == 1:
                    print(f"Example: predicted for protein {identifier} with length {s_len}: {predictions[identifier]}")
                
    end = time.time()
    print('\n############# STATS #############')
    print('Total number of predictions: {}'.format(len(predictions)))
    print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format( 
            end-start, (end-start)/len(predictions), avg_length))
    print("Writing results now to disk ...")
    
    write_predictions(predictions,out_path)

    return True


def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=( 
            'embed.py creates ProstT5-Encoder embeddings for a given text '+
            ' file containing sequence(s) in FASTA-format.' +
            'Example: python predict_3Di.py --input /path/to/some_AA_sequences.fasta --output /path/to/some_3Di_sequences.fasta --half 1' ) )
    
    # Required positional argument
    parser.add_argument( '-i', '--input', required=True, type=str,
                    help='A path to a fasta-formatted text file containing protein sequence(s).')

    # Optional positional argument
    parser.add_argument( '-o', '--output', required=True, type=str, 
                    help='A path for saving the created embeddings as NumPy npz file.')

    
    # Required positional argument
    parser.add_argument('--model', required=False, type=str,
                    default="Rostlab/ProstT5_fp16",
                    help='Either a path to a directory holding the checkpoint for a pre-trained model or a huggingface repository link.' )

    # Optional argument
    parser.add_argument('--split_char', type=str, 
                    default='!',
                    help='The character for splitting the FASTA header in order to retrieve ' +
                        "the protein identifier. Should be used in conjunction with --id." +
                        "Default: '!' ")
    
    # Optional argument
    parser.add_argument('--id', type=int, 
                    default=0,
                    help='The index for the uniprot identifier field after splitting the ' +
                        "FASTA header after each symbole in ['|', '#', ':', ' ']." +
                        'Default: 0')

    parser.add_argument('--half', type=int, 
                    default=1,
                    help="Whether to use half_precision or not. Default: 1 (half-precision)")
    
    return parser

def main():
    parser     = create_arg_parser()
    args       = parser.parse_args()
    
    seq_path   = Path( args.input ) # path to input FASTAS
    out_path   = Path( args.output) # path where predictions should be written to
    model_dir  = args.model # path/repo_link to checkpoint
    
    if out_path.is_file():
        print("Output file is already existing and will be overwritten ...")
    
    split_char = args.split_char
    id_field   = args.id

    half_precision = False if int(args.half) == 0 else True
    assert not (half_precision and device=="cpu"), print("Running fp16 on CPU is not supported, yet")
    
    get_embeddings( 
        seq_path, 
        out_path, 
        model_dir, 
        split_char, 
        id_field, 
        half_precision=half_precision,
        )


if __name__ == '__main__':
    main()
