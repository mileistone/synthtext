import sys
import random


def get_alpha_word(fp):
    raw_words = []
    with open(fp, 'r') as fd:
        for line in fd:
            segs = line.split()
            raw_words.extend(segs)
    raw_words = set(raw_words)
    alpha_words = []
    for word in raw_words:
        if word.isalpha():
            alpha_words.append(word)
    return list(alpha_words)


def shuffle_word_order(words):
    words_shuffle = []
    for word in words:
        word_list = list(word)
        random.shuffle(word_list)
        new_word = ''.join(word_list)
        words_shuffle.append(new_word)
    return words_shuffle


def gen_corpora(words, min_len, max_len, nlines):
    random.seed(0)
    lines = []
    for idx in range(nlines):
        tlen = min_len + int(random.random() * (max_len - min_len))
        sample = random.sample(words, tlen)
        line = ' '.join(sample)
        lines.append(line)
    return lines


def save2file(lines, fp):
    with open(fp, 'w') as fd:
        fd.writelines('\n'.join(lines))


if __name__ == '__main__':
    min_len = 5
    max_len = 13
    nlines = 10000
    fp = sys.argv[1]
    alpha_words = get_alpha_word(fp)
    #alpha_words_shuffle = shuffle_word_order(alpha_words)
    corpora = gen_corpora(alpha_words, min_len, max_len, nlines)
    save2file(corpora, 'coca_alpha_words.txt')
    #corpora = gen_corpora(alpha_words_shuffle, min_len, max_len, nlines)
    #save2file(corpora, 'alpha_words_shuffle')
