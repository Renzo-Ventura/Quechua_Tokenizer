# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 15:35:41 2021

@author: Rodolfo
"""

#! pip install tokenizers

from pathlib import Path

from tokenizers import normalizers
from tokenizers.normalizers import NFKC, Lowercase
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer


paths = [str(x) for x in Path("./corpus/").glob("**/*.txt")]


# Normalize corpus

bert_tokenizer = Tokenizer(WordPiece(unk_token= "[UNK]"))
bert_tokenizer.normalizer = normalizers.Sequence([NFKC(), Lowercase()])
bert_tokenizer.pre_tokenizer = Whitespace()

bert_tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)


# Training model

trainer = WordPieceTrainer(
    vocab_size=50265, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

bert_tokenizer.train(paths, trainer)
bert_tokenizer.save("quechuabert")

# Test model

output = bert_tokenizer.encode("allinllachu manan huk Perú! manan ñahuk")
print(output.tokens)



