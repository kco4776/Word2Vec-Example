import os
from gensim.models import Word2Vec
from collections import defaultdict
import numpy as np


def construct_weighted_embedding(embedding_fname, a=0.0001):
    dictionary = {}
    # load model
    words, vecs = load_word_embeddings("./word2vec/word2vec")
    # compute word frequency
    words_count, total_word_count = compute_word_frequency("./tokenized_data/corpus_mecab.txt")

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


def load_word_embeddings(vecs_fname):
    model = Word2Vec.load(vecs_fname)
    words = model.wv.index2word
    vecs = model.wv.vectors

    return words, vecs


def compute_word_frequency(embedding_corpus_fname):
    total_count = 0
    words_count = defaultdict(int)
    with open(embedding_corpus_fname, "r") as f:
        for line in f:
            tokens = line.strip().split()
            for token in tokens:
                words_count[token] += 1
                total_count += 1
    return words_count, total_count


def train_model(train_data_fname, model_fname):
    model = {"vectors": [], "labels": [], "sentences": []}
    train_data = self.load_or_tokenize_corpus(train_data_fname)
    with open(model_fname, "w") as f:
        for sentence, tokens, label in train_data:
            tokens = self.tokenizer.morphs(sentence)
            sentence_vector = self.get_sentence_vector(tokens)
            model["sentences"].append(sentence)
            model["vectors"].append(sentence_vector)
            model["labels"].append(label)
            str_vector = " ".join([str(el) for el in sentence_vector])
            f.writelines(sentence + "\u241E" + " ".join(tokens) + "\u241E" + str_vector + "\u241E" + label + "\n")
    return model


# construct weighted word embeddings
embeddings = construct_weighted_embedding("./word2vec/word2vec")
print("loading weighted embeddings, complete!")

# train model
print("train Continuous Bag of Words model")
model = train_model()
