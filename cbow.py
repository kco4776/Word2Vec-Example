import os
from gensim.models import Word2Vec

def load_or_construct_weighted_embedding(self, embedding_fname, embedding_method, embedding_corpus_fname, a=0.0001):
    dictionary = {}
    if os.path.exists(embedding_fname + "-weighted"):
        # load weighted word embeddings
        with open(embedding_fname + "-weighted", "r") as f2:
            for line in f2:
                word, weighted_vector = line.strip().split("\u241E")
                weighted_vector = [float(el) for el in weighted_vector.split()]
                dictionary[word] = weighted_vector
    else:
        # load pretrained word embeddings
        words, vecs = self.load_word_embeddings(embedding_fname, embedding_method)
        # compute word frequency
        words_count, total_word_count = self.compute_word_frequency(embedding_corpus_fname)
        # construct weighted word embeddings
        with open(embedding_fname + "-weighted", "w") as f3:
            for word, vec in zip(words, vecs):
                if word in words_count.keys():
                    word_prob = words_count[word] / total_word_count
                else:
                    word_prob = 0.0
                weighted_vector = (a / (word_prob + a)) * np.asarray(vec)
                dictionary[word] = weighted_vector
                f3.writelines(word + "\u241E" + " ".join([str(el) for el in weighted_vector]) + "\n")
    return dictionary


def load_word_embeddings(self, vecs_fname, method):
    if method == "word2vec":
        model = Word2Vec.load(vecs_fname)
        words = model.wv.index2word
        vecs = model.wv.vectors
    else:
        words, vecs = [], []
        with open(vecs_fname, 'r', encoding='utf-8') as f1:
            if "fasttext" in method:
                next(f1)  # skip head line
            for line in f1:
                if method == "swivel":
                    splited_line = line.replace("\n", "").strip().split("\t")
                else:
                    splited_line = line.replace("\n", "").strip().split(" ")
                words.append(splited_line[0])
                vec = [float(el) for el in splited_line[1:]]
                vecs.append(vec)
    return words, vecs


model = Word2Vec.load("./word2vec/word2vec")
words = model.wv.index2word
vecs = model.wv.vectors
