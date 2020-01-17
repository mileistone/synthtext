import numpy as np
import scipy.stats as sstat
import os.path as osp

from synthtext.config import load_cfg


class Corpora(object):
    """
    Provides text for words, paragraphs, sentences.
    """
    def __init__(self):
        """
        TXT_FP : path to file containing text data.
        """
        load_cfg(self)
        #fp = osp.join(self.data_dir, 'newsgroup/newsgroup.txt')
        #fp = osp.join(self.data_dir, 'newsgroup/alpha_words.txt')
        fp = osp.join(self.data_dir, 'newsgroup/alpha_words_shuffle.txt')
        self.fdict = {
            'WORD': self.sample_word,
            'LINE': self.sample_line,
            'PARA': self.sample_para
        }

        with open(fp, 'r') as f:
            self.txt = [l.strip() for l in f.readlines()]

    def check_symb_frac(self, txt, f=0.35):
        """
        T/F return : T iff fraction of symbol/special-charcters in
                     txt is less than or equal to f (default=0.25).
        """
        return np.sum([not ch.isalnum() for ch in txt]) / (len(txt) + 0.0) <= f

    def is_good(self, txt, f=0.35):
        """
        T/F return : T iff the lines in txt (a list of txt lines)
                     are "valid".
                     A given line l is valid iff:
                         1. It is not empty.
                         2. symbol_fraction > f
                         3. Has at-least self.min_nchar characters
                         4. Not all characters are i,x,0,O,-
        """
        def is_txt(l):
            char_ex = ['i', 'I', 'o', 'O', '0', '-']
            chs = [ch in char_ex for ch in l]
            return not np.all(chs)

        return [(len(l) > self.min_nchar and self.check_symb_frac(l, f)
                 and is_txt(l)) for l in txt]

    def center_align(self, lines):
        """
        PADS lines with space to center align them
        lines : list of text-lines.
        """
        ls = [len(l) for l in lines]
        max_l = max(ls)
        for i in range(len(lines)):
            l = lines[i].strip()
            dl = max_l - ls[i]
            lspace = dl // 2
            rspace = dl - lspace
            lines[i] = ' ' * lspace + l + ' ' * rspace
        return lines

    def get_lines(self, nline, nword, nchar_max, f=0.35, niter=100):
        def h_lines(niter=100):
            lines = ['']
            iter_ = 0
            while not np.all(self.is_good(lines, f)) and iter_ < niter:
                iter_ += 1
                line_start = np.random.choice(len(self.txt) - nline)
                lines = [self.txt[line_start + i] for i in range(nline)]
            return lines

        lines = ['']
        iter_ = 0
        while not np.all(self.is_good(lines, f)) and iter_ < niter:
            iter_ += 1
            lines = h_lines(niter=100)
            # get words per line:
            nline = len(lines)
            for i in range(nline):
                words = lines[i].split()
                dw = len(words) - nword[i]
                if dw > 0:
                    first_word_index = np.random.choice(range(dw + 1))
                    lines[i] = ' '.join(
                        words[first_word_index:first_word_index + nword[i]])

                while len(
                        lines[i]) > nchar_max:  #chop-off characters from end:
                    if not np.any([ch.isspace() for ch in lines[i]]):
                        lines[i] = ''
                    else:
                        lines[i] = lines[i][:len(lines[i]) -
                                            lines[i][::-1].find(' ')].strip()

        if not np.all(self.is_good(lines, f)):
            return  #None
        else:
            return lines

    # main method
    def sample_text(self, nline_max, nchar_max, kind='WORD'):
        text = self.fdict[kind](nline_max, nchar_max)
        return text

    def sample_word(self, nline_max, nchar_max, niter=100):
        rand_line = self.txt[np.random.choice(len(self.txt))]
        words = rand_line.split()
        rand_word = np.random.choice(words)

        iter_ = 0
        while iter_ < niter and (not self.is_good([rand_word])[0]
                                 or len(rand_word) > nchar_max):
            rand_line = self.txt[np.random.choice(len(self.txt))]
            words = rand_line.split()
            rand_word = np.random.choice(words)
            iter_ += 1

        if not self.is_good([rand_word])[0] or len(rand_word) > nchar_max:
            return []
        else:
            return rand_word

    def sample_line(self, nline_max, nchar_max):
        nline = nline_max + 1
        while nline > nline_max:
            nline = np.random.choice([1, 2, 3], p=self.p_line_nline)

        # get number of words:
        nword = [
            self.p_line_nword[2] *
            sstat.beta.rvs(a=self.p_line_nword[0], b=self.p_line_nword[1])
            for _ in range(nline)
        ]
        nword = [max(1, int(np.ceil(n))) for n in nword]

        lines = self.get_lines(nline, nword, nchar_max, f=0.35)
        if lines is not None:
            return '\n'.join(lines)
        else:
            return []

    def sample_para(self, nline_max, nchar_max):
        # get number of lines in the paragraph:
        nline = nline_max * sstat.beta.rvs(a=self.p_para_nline[0],
                                           b=self.p_para_nline[1])
        nline = max(1, int(np.ceil(nline)))

        # get number of words:
        nword = [
            self.p_para_nword[2] *
            sstat.beta.rvs(a=self.p_para_nword[0], b=self.p_para_nword[1])
            for _ in range(nline)
        ]
        nword = [max(1, int(np.ceil(n))) for n in nword]

        lines = self.get_lines(nline, nword, nchar_max, f=0.35)
        if lines is not None:
            # center align the paragraph-text:
            if np.random.rand() < self.center_para:
                lines = self.center_align(lines)
            return '\n'.join(lines)
        else:
            return []
