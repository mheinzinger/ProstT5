import sys
import os
from Bio import SeqIO

in_fasta = sys.argv[1]
in_3di = sys.argv[2]
db_name = sys.argv[3]

# read in amino-acid sequences
sequences_aa = {}
for record in SeqIO.parse(in_fasta, "fasta"):
    sequences_aa[record.id] = str(record.seq)

# read in 3Di strings
sequences_3di = {}
for record in SeqIO.parse(in_3di, "fasta"):
    if not record.id in sequences_aa.keys():
        print("Warning: ignoring 3Di entry {}, since it is not in the amino-acid FASTA file".format(record.id))
    else:
        sequences_3di[record.id] = str(record.seq).upper()

# assert that we parsed 3Di strings for all sequences in the amino-acid FASTA file
for id in sequences_aa.keys():
    if not id in sequences_3di.keys():
        print("Error: entry {} in amino-acid FASTA file has no corresponding 3Di string".format(id))
        quit()

# generate TSV file contents
tsv_aa = ""
tsv_3di = ""
tsv_header = ""
for i,id in enumerate(sequences_aa.keys()):
    tsv_aa += "{}\t{}\n".format(str(i+1), sequences_aa[id])
    tsv_3di += "{}\t{}\n".format(str(i+1), sequences_3di[id])
    tsv_header += "{}\t{}\n".format(str(i+1), id)

# write TSV files
with open("aa.tsv", "w") as f:
    f.write(tsv_aa)
with open("3di.tsv", "w") as f:
    f.write(tsv_3di)
with open("header.tsv", "w") as f:
    f.write(tsv_header)

# create Foldseek database
os.system("foldseek tsv2db aa.tsv {} --output-dbtype 0".format(db_name))
os.system("foldseek tsv2db 3di.tsv {}_ss --output-dbtype 0".format(db_name))
os.system("foldseek tsv2db header.tsv {}_h --output-dbtype 12".format(db_name))

# clean up
os.remove("aa.tsv")
os.remove("3di.tsv")
os.remove("header.tsv")
