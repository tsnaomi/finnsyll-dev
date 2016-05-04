# coding=utf-8

import phonology as phon

from compound import FinnSeg, finn


class LOOCV:

    def __init__(self, approach=None, dev=True):
        if dev:
            TRAINING = finn.full_training_set()
            VALIDATION = [t.id for t in finn.dev_set()]

        else:
            raise ValueError()
            TRAINING = finn.all_data()
            VALIDATION = [t.id for t in finn.test_set()]

        if approach == 'Baseline':
            segment = lambda t, F=None: t.orth.lower()

        else:
            segment = lambda t, F: F.segment(t.orth)

        RESULTS = {'TP': [], 'FP': [], 'TN': [], 'FN': [], 'bad': []}

        for t in VALIDATION:
            training = TRAINING.filter(finn.Token.id != t)
            t = finn.Token.query.get(t)

            F = FinnSeg(
                training=training,
                filename='data/loocv/' + str(t),
                approach=approach,
                Eval=False,
                )

            word = segment(t, F)
            gold = phon.replace_umlauts(t.gold_base, put_back=True)
            label = F._label(word, gold, t.is_complex)
            RESULTS[label].append((word, gold, F.get_morphemes(t.orth)))

        F._evaluate(RESULTS)

        self.description = F.report


if __name__ == '__main__':
    LOOCV()
    # LOOCV(approach='Unviolable')
    # LOOCV(approach='OT')
