# coding=utf-8

from compound import FinnSeg, finn


class CrossValidation:

    def __init__(self, approach=None):
        total = finn.full_training_set().count()
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.accuracy = 0.0

        for fold in range(1, 10):
            # print 'Fold ', fold

            TRAINING = finn.Token.query.filter(finn.Token.fold != fold)
            VALIDATION = finn.Token.query.filter_by(fold=fold)
            weight = float(VALIDATION.count()) / total

            F = FinnSeg(
                training=TRAINING,
                validation=VALIDATION,
                filename='data/morfessor-fold' + str(fold),
                Print=False,
                approach=approach,
                )

            self.precision += F.w_precision * weight
            self.recall += F.w_recall * weight
            self.f1 += F.w_f1 * weight
            self.accuracy += F.w_accuracy * weight

        self.description = (
            '\n'
            '---- Cross Validation: FinnSeg ----------------------------------'
            '\n\n%s'
            '\n\nWord-Level:'
            '\n\tP/R:\t%s / %s\n\tF1:\t%s\n\tAcc.:\t%s'
            '\n\n'
            '-----------------------------------------------------------------'
            '\n'
            ) % (
                F.description,
                self.precision, self.recall, self.f1, self.accuracy,
                )

        print self.description

if __name__ == '__main__':
    CrossValidation()
    CrossValidation(approach='Unviolable')
    CrossValidation(approach='OT')
