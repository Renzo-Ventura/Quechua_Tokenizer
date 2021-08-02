#! pip install git+https://github.com/huggingface/transformers#egg=transformers[sentencepiece]
#! pip install datasets
from pathlib import Path
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from datasets import load_dataset

text = "corpus/quz_corpus.txt"

dataset = load_dataset("text", data_files={"train": text, "validation": text}, split="train")

batch_size = 1000
def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]

tokenizer = Tokenizer(models.BPE())

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(vocab_size=50265, special_tokens=["<|endoftext|>"])

tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()

encoding = tokenizer.encode("allinllachu manan allinlla huk wasipita")
print(encoding.tokens)

tokenizer.save("tokenizerbpe.json")