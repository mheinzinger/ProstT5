#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:27:44 2023

@author: mheinzinger
"""

import argparse
import time
from pathlib import Path
import torch
import h5py
from transformers import T5EncoderModel, T5Tokenizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))


def get_T5_model(t5_dir):
    print("Loading T5 from: {}".format(t5_dir))
    model = T5EncoderModel.from_pretrained(t5_dir).to(device)
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(t5_dir, do_lower_case=False )
    return model, vocab


def read_fasta( fasta_path, split_char, id_field, is_3Di ):
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
                # repl. all white-space chars and join seqs spanning multiple lines
                if is_3Di:
                    sequences[ uniprot_id ] += ''.join( line.split() ).replace("-","").lower() # drop gaps and cast to upper-case
                else:
                    sequences[ uniprot_id ] += ''.join( line.split() ).replace("-","")
                    
    return sequences


def get_embeddings( seq_path, emb_path, model_dir, split_char, id_field, 
                       per_protein, half_precision, is_3Di,
                       max_residues=4000, max_seq_len=1000, max_batch=100 ):
    
    seq_dict = dict()
    emb_dict = dict()

    # Read in fasta
    seq_dict = read_fasta( seq_path, split_char, id_field, is_3Di )
    prefix = "<fold2AA>" if is_3Di else "<AA2fold>"
    
    model, vocab = get_T5_model(model_dir)
    if half_precision:
        model = model.half()
        print("Using model in half-precision!")

    print('########################################')
    print(f"Input is 3Di: {is_3Di}")
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
            
            # batch-size x seq_len x embedding_dim
            # extra token is added at the end of the seq
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                # account for prefix in offset
                emb = embedding_repr.last_hidden_state[batch_idx,1:s_len+1]
                
                if per_protein:
                    emb = emb.mean(dim=0)
                emb_dict[ identifier ] = emb.detach().cpu().numpy().squeeze()
                if len(emb_dict) == 1:
                    print("Example: embedded protein {} with length {} to emb. of shape: {}".format(
                                identifier, s_len, emb.shape))

    end = time.time()
    
    with h5py.File(str(emb_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            # noinspection PyUnboundLocalVariable
            hf.create_dataset(sequence_id, data=embedding)

    print('\n############# STATS #############')
    print('Total number of embeddings: {}'.format(len(emb_dict)))
    print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format( 
            end-start, (end-start)/len(emb_dict), avg_length))
    return True


def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=( 
            'prostT5_embedder.py creates ProstT5 embeddings for a given text '+
            ' file containing sequence(s) in FASTA-format.') )
    
    # Required positional argument
    parser.add_argument( '-i', '--input', required=True, type=str,
                    help='A path to a fasta-formatted text file containing protein sequence(s).')

    # Optional positional argument
    parser.add_argument( '-o', '--output', required=True, type=str, 
                    help='A path for saving the created embeddings as NumPy npz file.')

    
    # Required positional argument
    parser.add_argument('--model', required=False, type=str,
                    default="/mnt/home/mheinzinger/deepppi1tb/language_models/T5_xl_uniref50_v3/",
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
    # Optional argument
    parser.add_argument('--per_protein', type=int, 
                    default=0,
                    help="Whether to return per-residue embeddings (0: default) or the mean-pooled per-protein representation (1).")
        
    parser.add_argument('--half', type=int, 
                    default=0,
                    help="Whether to use half_precision or not. Default: 0 (full-precision)")
    
    parser.add_argument('--is_3Di', type=int, 
                    default=0,
                    help="Whether to create embeddings for 3Di or AA file. Default: 0 (generate AA-embeddings)")
    
    return parser

def main():
    parser     = create_arg_parser()
    args       = parser.parse_args()
    
    seq_path   = Path( args.input ) # path to input FASTAS
    emb_path   = Path( args.output) # path where embeddings should be stored
    model_dir  = Path( args.model ) # path/repo_link to checkpoint
    
    split_char = args.split_char
    id_field   = args.id

    per_protein    = False if int(args.per_protein) == 0 else True
    half_precision = False if int(args.half)        == 0 else True
    is_3Di         = False if int(args.is_3Di)      == 0 else True


    get_embeddings( 
        seq_path, 
        emb_path, 
        model_dir, 
        split_char, 
        id_field, 
        per_protein=per_protein,
        half_precision=half_precision, 
        is_3Di=is_3Di 
        )


if __name__ == '__main__':
    main()
