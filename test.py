from gensim.models import Word2Vec


def load_word_embeddings(vecs_fname):
    print("loading word embeddings...")
    print(f"vecs_fname:{vecs_fname}")
    embedding_model = Word2Vec.load(vecs_fname)
    print("loading complete!")
    words = embedding_model.wv.index2word
    vecs = embedding_model.wv.vectors
    return words, vecs