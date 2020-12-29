import os
from gensim.models import Word2Vec
from collections import defaultdict
import numpy as np
from tokenize import morphs


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


def load_tokenized_corpus(train_data_fname):
    data = []
    with open(train_data_fname, 'r') as f:
        for line in f:
            sentence, tokens, label = line.strip().split("\u241E")
            data.append([sentence, tokens.split(), label])
    return data


def get_sentence_vector(tokens, dim=100):
    vector = np.zeros(dim)
    for token in tokens:
        if token in embeddings.keys():
            vector += embeddings[token]
    vector /= len(tokens)
    vector_norm = np.linalg.norm(vector)
    if vector_norm != 0:
        unit_vector = vector / vector_norm
    else:
        unit_vector = np.zeros(dim)
    return unit_vector


def train_model(model_fname):
    model = {"vectors": [], "labels": [], "sentences": []}
    train_data = load_tokenized_corpus("./tokenized_data/ratings_train_mecab.txt")
    with open(model_fname, "w") as f:
        for sentence, tokens, label in train_data:
            tokens = morphs(sentence)
            sentence_vector = get_sentence_vector(tokens)
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
full_path = "./cbow/word2vec"
model_path = '/'.join(full_path.split('/')[:-1])
if not os.path.exists(model_path):
    os.makedirs(model_path)
model = train_model(full_path)
