from gensim.models import Word2Vec
import logging
import os


class Word2VecCorpus:
    def __init__(self, corpus_fname):
        self.corpus_fname = corpus_fname

    def __iter__(self):
        with open(self.corpus_fname, 'r', encoding='utf-8') as f:
            for sentence in f:
                tokens = sentence.replace('\n', '').strip().split(" ")
                yield tokens


logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
full_path = "C:/Users/kco47/Desktop/pythonProject/word2vec/word2vec"
model_path = "/".join(full_path.split("/")[:-1])
if not os.path.exists(model_path):
    os.makedirs(model_path)
corpus = Word2VecCorpus("./tokenized_data/corpus_mecab.txt")
model = Word2Vec(corpus, size=100, workers=4, sg=1)
model.save(full_path)
