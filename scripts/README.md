The scripts in this folder simplify usabliity of ProstT5.

- `embed.py` allows to derive embeddings either from amino acid- or 3Di-sequences. The following command generates amino acid embeddings:
  
  `python embed.py --input /path/to/some_sequences.fasta --output /path/to/some_embeddings.h5 --half 1 --is_3Di 0`

  If you want to derive per-protein embeddings for 3Di sequences, you need to adjust the following:

  `python embed.py --input /path/to/some_sequences.fasta --output /path/to/some_embeddings.h5 --half 1 --is_3Di 1 --per_protein 1`

- `translate.py` allows to translate from sequence to structure or vice versa. In order to translate from amino acids to 3Di, you need to run the following:
  
  `python translate_clean.py --input /path/to/some_AA_sequences.fasta --output /path/to/output_directory --half 1 --is_3Di 0`

  If you want to translate from 3Di to AA (inverse folding), you need to run the following:

    `python translate_clean.py --input /path/to/some_3Di_sequences.fasta --output /path/to/output_directory --half 1 --is_3Di 1`

- `predict_3Di_encoderOnly.py` takes input:FASTA(AA) to output:FASTA(3Di) using a CNN trained on top of embeddings from ProstT5's Encoder:
  
  `python predict_3Di.py --input /path/to/some_AA_sequences.fasta --output /path/to/some_3Di_sequences.fasta --half 1`
  
  If predicted 3Di will only be used for remote homology detection via Foldseek, the generation capability of ProstT5's decoder can be dropped for a substantial speed-up.

  This makes it easier to scale 3Di-prediction to metagenomic datasets.


In all cases, you can adjust the [batch-size](https://github.com/mheinzinger/ProstT5/blob/main/scripts/translate.py#L126)/speed according to your available (GPU) resources.
If you have batch_size>1 for translate.py, there can be minimal [length mismatches](https://github.com/mheinzinger/ProstT5/blob/main/scripts/translate.py#L218) between source and target. To ensure identical length, set batch_size=1 for translate.py. This does not affect embedding generation as embeddings are solely derived from the Encoder.
