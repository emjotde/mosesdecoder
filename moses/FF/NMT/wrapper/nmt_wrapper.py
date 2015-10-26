import numpy
import cPickle
import sys
import time

from encdec import parse_input, RNNEncoderDecoder
from state import prototype_state
from collections import defaultdict


class Timer(object):
    def __init__(self):
        self.parts = defaultdict(list)
        self.starts = defaultdict(int)

    def start(self, name):
        if self.starts[name] == 0:
            self.starts[name] = time.time()

    def finish(self, name):
        if self.starts[name] != 0:
            p = time.time() - self.starts[name]
            self.parts[name].append(p)
            self.starts[name] = 0
        else:
            pass

    def print_stats(self):
        for kk, vv in self.parts.iteritems():
            print >> sys.stderr, "{}: {}".format(kk, sum(vv))
            sys.stderr.flush()

    def __del__(self):
        self.print_stats()


class NMTWrapper(object):
    """ NMT Wrapper """
    def __init__(self, state_path, model_path, source_vocab_path,
                 target_vocab_path):
        self.state_path = state_path
        self.model_path = model_path
        self.source_vocab_path = source_vocab_path
        self.target_vocab_path = target_vocab_path
        self.timer = Timer()

    def build(self):

        self.state = prototype_state()
        with open(self.state_path) as src:
            self.state.update(cPickle.load(src))
        self.state['indx_word_target'] = None
        self.state['indx_word'] = None
        self.state['word_indx'] = self.source_vocab_path
        self.state['word_indx_trgt'] = self.target_vocab_path

        self.source_vocab = cPickle.load(open(self.state['word_indx']))
        self.target_vocab = cPickle.load(open(self.state['word_indx_trgt']))

        self.target_vocab["</s>"] = 30000

        rng = numpy.random.RandomState(self.state['seed'])
        self.enc_dec = RNNEncoderDecoder(self.state, rng, skip_init=True)
        self.enc_dec.build()
        self.lm_model = self.enc_dec.create_lm_model()
        self.lm_model.load(self.model_path)
        self.state = self.enc_dec.state

        self._compile()

        self.eos_id = self.state['null_sym_target']
        self.unk_id = self.state['unk_sym_target']

    def _compile(self):
        self.comp_repr = self.enc_dec.create_representation_computer()
        self.comp_init_states = self.enc_dec.create_initializers()
        self.comp_next_probs = self.enc_dec.create_next_probs_computer()
        self.comp_next_states = self.enc_dec.create_next_states_computer()

    def get_unk(self, words):
        unks = [1 if word in self.target_vocab.keys() else 0 for word in words]
        return unks

    def get_context_vector(self, source_sentence):
        seq = parse_input(self.state, self.source_vocab, source_sentence)
        c = self.comp_repr(seq)[0]
        return c

    def get_log_prob(self, next_word, c, last_word="", state=None):
        return self.get_log_probs([next_word], c, last_word, state)

    def get_log_probs(self, next_words, c, last_word="", state=None):
        cumulated_score = 0.0
        if not last_word:
            last_word = numpy.zeros(1, dtype="int64")
        else:
            last_word = [self.target_vocab.get(last_word, self.unk_id)]

        if state is None:
            states = map(lambda x: x[None, :], self.comp_init_states(c))
        else:
            states = [state]

        for next_word in next_words:
            next_indx = self.target_vocab.get(next_word, self.unk_id)

            log_probs = numpy.log(self.comp_next_probs(c, 0, last_word,
                                                       *states)[0])
            cumulated_score += log_probs[0][next_indx]

            voc_size = log_probs.shape[1]
            word_indices = numpy.array([next_indx]) % voc_size

            new_states = self.comp_next_states(c, 0, word_indices, *states)
            last_word = word_indices
            states = [new_states[0]]

        return cumulated_score, new_states[0]

    def get_vec_log_probs(self, next_words, c, last_words=[], states=[]):
        if len(last_words) == 0:
            phrase_num = 1
        else:
            phrase_num = len(last_words)
        if len(last_words) == 1 and len(last_words[0]) == 0:
            last_words = numpy.zeros(phrase_num)
            print last_words.astype("int64")
        else:
            tmp = []
            for last_word in last_words:
                if last_word == "":
                    tmp.append(0)
                else:
                    tmp.append(self.target_vocab.get(last_word,
                                                     self.unk_id))
            last_words = numpy.array(tmp, dtype="int64")

        if len(states) == 0:
            states = [numpy.repeat(self.comp_init_states(c)[0][numpy.newaxis, :], phrase_num, 0)]
        else:
            states = [numpy.concatenate(states)]

        next_indxs = [self.target_vocab.get(next_word, self.unk_id)
                      for next_word in next_words]

        log_probs = numpy.log(self.comp_next_probs(c, 0, last_words.astype("int64"),
                                                   *states)[0])
        cumulated_score = [log_probs[i][next_indxs].tolist()
                           for i in range(phrase_num)]

        new_states = []
        for val in next_indxs:
            intmp = [val] * phrase_num
            new_states.append(numpy.split(self.comp_next_states(c, 0, intmp, *states)[0], phrase_num))

        return cumulated_score, new_states, self.get_unk(next_words)

    def get_next_states(self, next_words, c, states):
        states = [numpy.concatenate(states)]
        next_indxs = [self.target_vocab.get(next_word, self.unk_id)
                      for next_word in next_words]
        return numpy.split(self.comp_next_states(c, 0, next_indxs, *states)[0])

    def get_log_prob_states(self, next_words, c, last_words=[], states=[]):
        self.timer.start('get_log_prob_states')

        if len(last_words) == 0:
            phrase_num = 1
        else:
            phrase_num = len(last_words)

        if len(last_words) >= 1 and len(last_words[0]) == 0:
            last_words = numpy.zeros(phrase_num)
        else:
            tmp = []
            for last_word in last_words:
                if last_word == "":
                    tmp.append(0)
                else:
                    tmp.append(self.target_vocab.get(last_word, self.unk_id))
            last_words = numpy.array(tmp)
            last_words = last_words.astype("int64")
        if len(states) == 0:
            states = [numpy.repeat(self.comp_init_states(c)[0][numpy.newaxis, :], phrase_num, 0)]
        else:
            states = [numpy.concatenate(states)]

        next_indxs = [self.target_vocab.get(next_word, self.unk_id)
                      for next_word in next_words]
        self.timer.start('get_log_prob_states_probs')
        probs = self.comp_next_probs(c, 0, last_words.astype("int64"),
                                     *states)[0]
        self.timer.finish('get_log_prob_states_probs')
        self.timer.start('get_log_prob_states_states')
        log_probs = numpy.log(probs)
        cumulated_score = [log_probs[i][next_indxs[i]]
                           for i in range(phrase_num)]

        self.timer.start('get_log_prob_states_states')
        new_states = self.comp_next_states(c, 0, next_indxs, *states)[0]
        self.timer.finish('get_log_prob_states_states')
        new_states = numpy.split(new_states, phrase_num)
        self.timer.finish('get_log_prob_states')

        return cumulated_score, new_states, self.get_unk(next_words)

    def get_nbest_list(self, state):
        return None
