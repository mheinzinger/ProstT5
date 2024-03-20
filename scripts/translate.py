#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 21:59:01 2023

@author: mheinzinger
"""


import argparse
import time
from pathlib import Path

from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, set_seed
set_seed(42) # ensure reproducability
import torch
if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print("Using device: {}".format(device))

import json

# If you want to experiment with your own generation config, add it here
class GenConfigs():
    def __init__(self):
        self.configs = self.get_configs()
    
    def get_configs(self):
        configs={}
        configs["aa2ss"] = self.aa2ss()
        configs["ss2aa"] = self.ss2aa()
        return configs
    
    def aa2ss(self):
        return { 
                  "do_sample": True,
                  "num_beams": 3, 
                  "top_p" : 0.95, 
                  "temperature" : 1.2, 
                  "top_k" : 6,
                  "repetition_penalty" : 1.2,
                }
    
    def ss2aa(self):
        return {
            "do_sample": True,
            "top_p" : 0.85,
            "temperature" : 1.0,
            "top_k" : 3,
            "repetition_penalty" : 1.2,
            }


def get_T5_model(model_dir,half_precision):
    print("##########################")
    print(f"Loading model from: {model_dir}")
    print("##########################")
    model=AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model=model.eval()
    if half_precision:
        model=model.half()
    tokenizer= T5Tokenizer.from_pretrained(model_dir)
    return model, tokenizer


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
                sequences[ uniprot_id ] = ''
            else:
                # repl. all white-space chars and join seqs spanning multiple lines
                if is_3Di:
                    sequences[ uniprot_id ] += ''.join( line.split() ).replace("-","").lower()
                else:
                    sequences[ uniprot_id ] += ''.join( line.split() ).replace("-","")
    
    example = sequences[uniprot_id]
        
    print("##########################")
    print(f"Input is 3Di: {is_3Di} . Sequence below should be lower-case if input is 3Di.")
    print(f"Example sequence: >{uniprot_id}\n{example}")
    print("##########################")
    
    return sequences


def write_fasta(out_path,results,prefix):
    seq_out_p = out_path / f"{prefix}_sequences.fasta"
    with open(seq_out_p,"w") as out_f: # write sequences
        out_f.write( "\n".join( [ f">{seq_id}\n{seq}" 
                                 for seq_id, seq in results.items() 
                                 ] ) )
    return None

def write_config(out_path,gen_kwargs):
    with open(out_path/"gen_config.json","w") as out_f:
        json.dump(gen_kwargs, out_f, ensure_ascii=False)
    return None

def translate(in_path, 
              out_path, 
              model_dir, 
              split_char, 
              id_field, 
              half_precision, 
              is_3Di, 
              gen_kwargs,
              # the number of sequences you want to generate per input sequence
              # Similar to batching (below) but for each input sequence individually
              # this can lead to RunTimeError/OOM if set too high
              num_return_sequences=1, 
              # Batch parameters. This increases number of input sequences processed in parallel.
              # set max_batch=1 if you need to ensure that sequences are of identical length
              # otherwise, there can be minimal length mismatch for border-line cases
              max_residues=4000, max_seq_len=1000, max_batch=1,
              ):
    
    
    model, tokenizer = get_T5_model(model_dir, half_precision)
    
    seq_dict = read_fasta( in_path, split_char, id_field, is_3Di )
    seq_dict = { s_id : s for s_id, s in seq_dict.items() }
            
    if is_3Di: # if we go from 3Di (start/s) to AA (target/t) 
        prefix_s2t = "<fold2AA>"
        # don't generate 3Di or rare/ambig. AAs when outputting AA
        noGood = "acdefghiklmnpqrstvwyXBZ"

    else: # if we go from AA (start) to 3Di (target)
        prefix_s2t = "<AA2fold>"
        # don't generate AAs when outputting 3Di
        noGood = "ARNDBCEQZGHILKMFPSTWYVXOU"
        
    bad_words = tokenizer( [" ".join(list(noGood))], 
                          add_special_tokens=False
                          ).input_ids


    avg_length = int(sum([ len(seq) for _, seq in seq_dict.items()]) / len(seq_dict))
    # start with longest sequences to quickly debug OOM
    seq_dict   = sorted( seq_dict.items(), key=lambda kv: len( seq_dict[kv[0]] ), reverse=True )
    
    print("Average sequence length: {} measured over {} sequences".format(avg_length, len(seq_dict)))
    print("Parameters used for generation: {}".format(gen_kwargs))
    
    batch = list()
    generation_results = {}
    
    start = time.time()
    for seq_idx, (fasta_id, seq) in enumerate(seq_dict,1): # for each sequence in the FASTA file
        seq_len = len(seq)
        seq = seq.replace('U','X').replace('Z','X').replace('O','X').replace("B","X")
        seq = " ".join(list(seq))
        batch.append((fasta_id,seq,seq_len))
        
        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed 
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len 
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            ids, seqs, seq_lens = zip(*batch)
            batch = list()
            max_len=int( max(seq_lens) + 1) 
            min_len=int( min(seq_lens) + 1)

            # starting point tokens
            start_encoding = tokenizer.batch_encode_plus( [prefix_s2t + " " + s for s in seqs], 
                                       add_special_tokens=True,
                                       padding="longest",
                                       return_tensors='pt' 
                                       ).to(device)
            
            try:
                with torch.no_grad():
                    # forward translation tokens
                    target = model.generate( 
                        start_encoding.input_ids, 
                        attention_mask=start_encoding.attention_mask, 
                        max_length=max_len, # max length of generated text
                        min_length=min_len, # minimum length of the generated text
                        early_stopping=True, # stop early if end-of-text token is generated
                        length_penalty=1.0, # import for correct normalization of scores
                        bad_words_ids=bad_words, # avoid generation of tokens from other vocabulary
                        num_return_sequences=num_return_sequences, # return only a single sequence
                        **gen_kwargs
                        )
            except RuntimeError as e:
                # rare cases trigger the following error (seemed to depend on generation config):
                # RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
                print(f"RuntimeError during target generation for {fasta_id}")
                print("If this is triggered by OOM, try lowering num_return_sequences and/or max_batch")
                print(e)
                continue
            
            t_strings = tokenizer.batch_decode( target, skip_special_tokens=True )

            for batch_idx, identifier in enumerate(ids): # all individual input sequences in a batch
                for seq_idx in range(num_return_sequences): # all sequences generated per individual input sequence
                    out_id = f"{seq_idx}_" + identifier
                    s_len = seq_lens[batch_idx]
                    # offset accounts for multiple sequences generated per input sequence 
                    batch_seq_idx = (batch_idx*num_return_sequences) + seq_idx
                    t_seq = "".join( t_strings[batch_seq_idx].split(" ")) # target sequence (prediction)
                    t_len = len(t_seq)
                    
                    # this is only triggered rarely and only if processing in batched mode, 
                    # happens esp. for proteins with L>512 (beyond training length)
                    if t_len!=s_len:
                        print(f"source length={s_len} vs target length={t_len}")
                        if t_len>s_len: # truncate if target longer than groundtruth
                            t_seq = t_seq[:s_len]
                        elif s_len<t_len:
                            while t_len<s_len:
                                t_seq+="d" # append d's in case of too short
                                t_len=len(t_seq)
                    
                    generation_results[out_id] = t_seq
                    if len(generation_results) == 1:
                        print(f"Example generation for {out_id}:\nseqs[batch_idx]\n{generation_results[out_id]}")

        
    compute_time = (time.time() - start)
    print(f"Translating {len(generation_results)} proteins with an avg. length of {avg_length} "+
          f" took {compute_time/60:.1f}[m] ({compute_time/len(generation_results):.1f}[s/protein])\n"+
          "Writing results ..."
          )
    write_fasta(out_path,generation_results,"generated")
    if len(gen_kwargs)>0:
        write_config(out_path,gen_kwargs)
    return None

def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=( 
            'translate.py uses ProstT5 to translate from sequence (AA) to structure (3Di) '+
            'and vice versa. Expects a fasta-formatted file as input. Example: ' + 
            'python translate.py --input /some/file.fasta --output /some/translation_output_dir') )
    
    # Input file in FASTA format. Can either have 3Di sequences (lower-case!) or AA seqs (upper case)
    parser.add_argument( '-i', '--input', required=True, type=str,
                    help='A path to a fasta-formatted text file containing protein sequence(s).')

    # Output directory where to write results to
    parser.add_argument( '-o', '--output', required=True, type=str, 
                    help='Output directory to write generated sequences and config to.')
    
    # Required positional argument
    parser.add_argument('--model', required=False, type=str,
                    default="Rostlab/ProstT5",
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
                    default=0,
                    help="Whether to use half_precision or not. Default: 0 (full-precision)")
    
    parser.add_argument('--is_3Di', type=int, 
                    default=0,
                    help="Whether the input is 3Di or AA file. Default: 0 (--> input is AA so translate from AA to 3Di)")
    return parser


def main():
    parser     = create_arg_parser()
    args       = parser.parse_args()
    
    in_path   = Path( args.input ) # FASTA-input file with AAs or 3Di
    out_path  = Path( args.output) # Output directory to write results to
    if out_path.is_dir():
        print("Result directory already exists! - Watch out to not overwriting existing results!")
    else:
        out_path.mkdir()

    model_dir = Path( args.model ) # path/repo_link to checkpoint
    
    split_char = args.split_char
    id_field   = args.id

    half_precision = False if int(args.half)   == 0 else True
    is_3Di         = False if int(args.is_3Di) == 0 else True
    
    print(f"is_3Di is {is_3Di}. (0=expect input to be AA, 1= input is 3Di")
    gen_configs = GenConfigs()
    # if input is 3Di use config to translate from 3Di to AA (ss2aa) 
    gen_config = gen_configs.configs["ss2aa"]  if is_3Di else gen_configs.configs["aa2ss"]
    
    translate( 
              in_path, 
              out_path, 
              model_dir,
              split_char, 
              id_field, 
              half_precision,
              is_3Di,
              gen_kwargs=gen_config
              )

    
if __name__=='__main__':
    main()
