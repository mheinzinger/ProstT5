# ProstT5
Bilingual Language Model for Protein Sequence and Structure

<br/>
<p align="center">
    <img width="95%" src="https://github.com/mheinzinger/ProstT5/blob/main/prostt5_sketch2.png" alt="ProstT5 training and inference sketch">
</p>
<br/>

[ProstT5](https://huggingface.co/Rostlab/ProstT5) (Protein structure-sequence T5) is a protein language model (pLM) which can translate between protein sequence and structure. 
It is based on [ProtT5-XL-U50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50), a T5 model trained on encoding protein sequences using span corruption applied on billions of protein sequences.
ProstT5 finetunes [ProtT5-XL-U50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) on translating between protein sequence and structure using 17M proteins with high-quality 3D structure predictions from the AlphaFoldDB.
Protein structure is converted from 3D to 1D using the 3Di-tokens introduced by [Foldseek](https://github.com/steineggerlab/foldseek).

<a name="quick"></a>
## 🚀&nbsp; Installation
ProstT5 is available via huggingface/transformers:
```console
pip install transformers
```
For more details, please follow the instructions for [transformers installations](https://huggingface.co/docs/transformers/installation).

<a name="quick"></a>
## 🚀&nbsp; Quick Start
Example for how to derive embeddings from ProstT5:
```python
from transformers import T5Tokenizer, T5EncoderModel
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False).to(device)

# Load the model
model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)

# only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
model.full() if device=='cpu' else model.half()

# prepare your protein sequences/structures as a list.
# Amino acid sequences are expected to be upper-case ("PRTEINO" below)
# while 3Di-sequences need to be lower-case ("strctr" below).
sequence_examples = ["PRTEINO", "strct"]

# replace all rare/ambiguous amino acids by X (3Di sequences do not have those) and introduce white-space between all sequences (AAs and 3Di)
sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

# The direction of the translation is indicated by two special tokens:
# if you go from AAs to 3Di (or if you want to embed AAs), you need to prepend "<AA2fold>"
# if you go from 3Di to AAs (or if you want to embed 3Di), you need to prepend "<fold2AA>"
sequence_examples = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s # this expects 3Di sequences to be already lower-case
                      for s in sequence_examples
                    ]

# tokenize sequences and pad up to the longest sequence in the batch
ids = tokenizer.batch_encode_plus(sequences_example,
                                  add_special_tokens=True,
                                  padding="longest",
                                  return_tensors='pt').to(device))

# generate embeddings
with torch.no_grad():
    embedding_rpr = model(
              ids.input_ids, 
              attention_mask=ids.attention_mask
              )

# extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens, incl. prefix ([0,1:8]) 
emb_0 = embedding_repr.last_hidden_state[0,1:8] # shape (7 x 1024)
# same for the second ([1,:]) sequence but taking into account different sequence lengths ([1,:6])
emb_1 = embedding_repr.last_hidden_state[1,1:6] # shape (5 x 1024)

# if you want to derive a single representation (per-protein embedding) for the whole protein
emb_0_per_protein = emb_0.mean(dim=0) # shape (1024)
```

Example for how to translate between sequence and structure (3Di) using ProstT5:
```python
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False).to(device)

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained("Rostlab/ProstT5").to(device)

# only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
model.full() if device=='cpu' else model.half()

# prepare your protein sequences/structures as a list.
# Amino acid sequences are expected to be upper-case ("PRTEINO" below)
# while 3Di-sequences need to be lower-case.
sequence_examples = ["PRTEINO", "SEQWENCE"]
min_len = min([ len(s) for s in folding_example])
max_len = max([ len(s) for s in folding_example])

# replace all rare/ambiguous amino acids by X (3Di sequences does not have those) and introduce white-space between all sequences (AAs and 3Di)
sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

# add pre-fixes accordingly. For the translation from AAs to 3Di, you need to prepend "<AA2fold>"
sequence_examples = [ "<AA2fold>" + " " + s for s in sequence_examples]

# tokenize sequences and pad up to the longest sequence in the batch
ids = tokenizer.batch_encode_plus(sequences_example,
                                  add_special_tokens=True,
                                  padding="longest",
                                  return_tensors='pt').to(device))

# Generation configuration for "folding" (AA-->3Di)
gen_kwargs_aa2fold = {
                  "do_sample": True,
                  "num_beams": 3, 
                  "top_p" : 0.95, 
                  "temperature" : 1.2, 
                  "top_k" : 6,
                  "repetition_penalty" : 1.2,
}

# translate from AA to 3Di (AA-->3Di)
with torch.no_grad():
  translations = model.generate( 
              ids.input_ids, 
              attention_mask=ids.attention_mask, 
              max_length=max_len, # max length of generated text
              min_length=min_len, # minimum length of the generated text
              early_stopping=True, # stop early if end-of-text token is generated
              num_return_sequences=1, # return only a single sequence
              **gen_kwargs_aa2fold
  )
# Decode and remove white-spaces between tokens
decoded_translations = tokenizer.batch_decode( translations, skip_special_tokens=True )
structure_sequences = [ "".join(ts.split(" ")) for ts in decoded_translations ] # predicted 3Di strings

# Now we can use the same model and invert the translation logic
# to generate an amino acid sequence from the predicted 3Di-sequence (3Di-->AA)

# add pre-fixes accordingly. For the translation from 3Di to AA (3Di-->AA), you need to prepend "<fold2AA>"
sequence_examples_backtranslation = [ "<fold2AA>" + " " + s for s in decoded_translations]

# tokenize sequences and pad up to the longest sequence in the batch
ids_backtranslation = tokenizer.batch_encode_plus(sequence_examples_backtranslation,
                                  add_special_tokens=True,
                                  padding="longest",
                                  return_tensors='pt').to(device))

# Example generation configuration for "inverse folding" (3Di-->AA)
gen_kwargs_fold2AA = {
            "do_sample": True,
            "top_p" : 0.85,
            "temperature" : 1.0,
            "top_k" : 3,
            "repetition_penalty" : 1.2,
}

# translate from 3Di to AA (3Di-->AA)
with torch.no_grad():
  backtranslations = model.generate( 
              ids_backtranslation.input_ids, 
              attention_mask=ids_backtranslation.attention_mask, 
              max_length=max_len, # max length of generated text
              min_length=min_len, # minimum length of the generated text
              #early_stopping=True, # stop early if end-of-text token is generated; only needed for beam-search
              num_return_sequences=1, # return only a single sequence
              **gen_kwargs_fold2AA
  )
# Decode and remove white-spaces between tokens
decoded_backtranslations = tokenizer.batch_decode( backtranslations, skip_special_tokens=True )
aminoAcid_sequences = [ "".join(ts.split(" ")) for ts in decoded_backtranslations ] # predicted amino acid strings

```

<a name="quick"></a>
## 💥&nbsp; Scripts and tutorials
Update: we now provide an example colab notebook showing how to run [inverse folding](https://github.com/mheinzinger/ProstT5/blob/main/notebooks/ProstT5_inverseFolding.ipynb) with ProstT5 as well as a script that allows to [extract embeddings](https://github.com/mheinzinger/ProstT5/blob/main/scripts/embed.py).

We will release other scripts that simplify embedding extraction and translation between sequence and structure asap.
In the meantime, you can easily modify existing [scripts](https://github.com/agemagician/ProtTrans/blob/master/Embedding/prott5_embedder.py) and [colab notebooks](https://colab.research.google.com/drive/1h7F5v5xkE_ly-1bTQSu-1xaLtTP2TnLF?usp=sharing) that explain how to extract embeddings from ProtT5 (only modifications needed: a) change model repository from ProtT5 to ProstT5, b) add prefixes as shown above accordingly and c) cast 3Di to lower-case.

<a name="foldseek"></a>
## 💥&nbsp; How to derive 3Di sequences from structures?
Structure strings (3Di sequences) as defined by [Foldseek](https://github.com/steineggerlab/foldseek) can be derived via the following commands (please, follow [installation instruction for Foldseek](https://github.com/steineggerlab/foldseek#installation) first):
```console
foldseek createdb directory_with_PDBs queryDB
foldseek lndb queryDB_h queryDB_ss_h
foldseek convert2fasta queryDB_ss queryDB_ss.fasta
```
This can be applied on a directory of PDB structures (can be experimental or predicted).
The 3Di-sequences can be used either to derive embeddings or can be used as starting point for inverse folding. 

Watch out that 3Di sequences output by Foldseek are by default upper-case while ProstT5 expects them to be lower-case to avoid tokenization clash with amino acids.

<a name="quick"></a>
## 📘&nbsp; Training data
We make our training data publicly available via [huggingface datasets](https://huggingface.co/datasets/adrianhenkel/lucidprots_full_data).
We will also make PDB files for our training data available on zenodo. 

<a name="quick"></a>
## 🚀&nbsp; Training scripts
For training, we used [this script](https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py) for pre-training on span-based denoising and [this script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization_no_trainer.py) for translation.

<a name="license"></a>
## 📘&nbsp; License
ProstT5 is released under the under terms of the [MIT license](https://choosealicense.com/licenses/mit/).

<a name="quick"></a>
## ✏️&nbsp; Citation
```
@ARTICLE
{Heinzinger2023.07.23.550085,
	author = {Michael Heinzinger and Konstantin Weissenow and Joaquin Gomez Sanchez and Adrian Henkel and Martin Steinegger and Burkhard Rost},
	title = {ProstT5: Bilingual Language Model for Protein Sequence and Structure},
	year = {2023},
	doi = {10.1101/2023.07.23.550085},
	journal = {bioRxiv}
}
```
