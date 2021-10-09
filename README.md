# T-SSKD

## Experiments
Dataset:
- IMDB 
- AMZR_p

================================ <br>
IMDB|AMZR_p <br>
BERT: 93.23 | 94.84 <br>
DitilBERT: 92.70 | 93.69 <br>
MiniLM: 92.96 | 94.77 <br>
Trans: 80.88 | 88.28 <br>
================================ <br>
T-SSKD (+sskd) <br>
: 91.19 (0, 01. 500, 16, 150, 42, 0.0003 0.1 temperature 1) <br>
: 91.68 (0, 01. 500, 16, 150, 42, 0.0003 0.1 temperature 2) <br>
T-SSKD (+sskd +attn) <br>
: 91.55 (fixed sskd | attn temperature 1) <br>
: 92.91 (fixed sskd | attn temperature 1, loss/N.head) <br>
: 93.43 (fixed sskd | attn temperature 1, loss/N.head, no vocab distribution, 57M->2048) <br>
