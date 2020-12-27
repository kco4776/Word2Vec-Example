import argparse
import re

import MeCab


def post_processing(tokens):
    results = []
    for token in tokens:
        # 숫자에 공백을 주어서 띄우기
        processed_token = [el for el in re.sub(r"(\d)", r" \1 ", token).split(" ") if len(el) > 0]
        results.extend(processed_token)
    return results


def morphs(sentence):
    tokenizer = MeCab.Tagger()
    tokens = []
    a = tokenizer.parseToNode(sentence)
    while a:
        if a.surface is not "":
            tokens.append(a.surface)
        a = a.next
    print(tokens)
    print("\n")
    return tokens


def tokenize(corpus_fname, output_fname, label=False):
    if pos:

    else:
        with open(corpus_fname, 'r', encoding='utf-8') as f1, open(output_fname, 'w', encoding='utf-8') as f2:
            for line in f1:
                sentence = line.replace('\n', '').strip()
                tokens = morphs(sentence)
                tokenized_sent = ' '.join(post_processing(tokens))
                f2.writelines(tokenized_sent + '\n')


def str2bool(str):
    return str.lower() in ["true", "t"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Location of input files')
    parser.add_argument('--output_path', type=str, help='Location of output files')
    parser.add_argument('--with_label', type=str, help='with label', default=False)
    args = parser.parse_args()
    tokenize(args.input_path, args.output_path, str2bool(args.with_label))
