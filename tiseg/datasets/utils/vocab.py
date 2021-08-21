import re

UNK_IDENTIFIER = '<unk>'  # <unk> is the word used to identify unknown words
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


def load_vocab(dict_file):
    with open(dict_file) as f:
        words = [w.strip() for w in f.readlines()]
    vocab_dict = {words[n]: n for n in range(len(words))}
    return vocab_dict


def load_indices_vocab(dict_file):
    with open(dict_file) as f:
        words = [w.strip() for w in f.readlines()]
    indices_vocab_dict = {n: words[n] for n in range(len(words))}
    return indices_vocab_dict


def sentence_to_vocab_indices(sentence, vocab_dict):
    words = SENTENCE_SPLIT_REGEX.split(sentence.strip())
    words = [w.lower() for w in words if len(w.strip()) > 0]
    # remove .
    if words[-1] == '.':
        words = words[:-1]
    vocab_indices = [
        (vocab_dict[w] if w in vocab_dict else vocab_dict[UNK_IDENTIFIER])
        for w in words
    ]
    return vocab_indices


PAD_IDENTIFIER = '<pad>'
EOS_IDENTIFIER = '<eos>'


def preprocess_sentence(sentence, vocab_dict, T):
    vocab_indices = sentence_to_vocab_indices(sentence, vocab_dict)
    # Append '<eos>' symbol to the end
    # vocab_indices.append(vocab_dict[EOS_IDENTIFIER])
    # Truncate long sentences
    if len(vocab_indices) > T:
        vocab_indices = vocab_indices[:T]
    # Pad short sentences at the beginning with the special symbol '<pad>'
    if len(vocab_indices) < T:
        vocab_indices = [vocab_dict[PAD_IDENTIFIER]
                         ] * (T - len(vocab_indices)) + vocab_indices
    return vocab_indices
