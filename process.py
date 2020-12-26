import argparse


def process_nsmc(corpus_path, output_fname, with_label=True):
    with open(corpus_path, 'r', encoding='utf-8') as f1, \
            open(output_fname, 'w', encoding='utf-8') as f2:
        next(f1)
        for line in f1:
            _, sentence, label = line.strip().split('\t')
            if not sentence: continue
            if with_label:
                f2.writelines(sentence + "\u241E" + label + "\n")
            else:
                f2.writelines(sentence + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess_mode', type=str, help='preprocess mode')
    parser.add_argument('--input_path', type=str, help='Location of input files')
    parser.add_argument('--output_path', type=str, help='Location of output files')
    parser.add_argument('--with_label', help='with label', type=str, default="False")
    args = parser.parse_args()

    if args.preprocess_mode == "nsmc":
        process_nsmc(args.input_path, args.output_path, args.with_label.lower() == "true")
    """elif args.preprocess_mode == "wiki":
        make_corpus(args.input_path, args.output_path)
    elif args.preprocess_mode == "korquad":
        process_korQuAD(args.input_path, args.output_path)
    elif args.preprocess_mode == "process-documents":
        process_documents(args.input_path, args.output_path)"""
