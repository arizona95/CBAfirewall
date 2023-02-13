from .algorithms import (
    M1Algorithm,
    M2Algorithm,
    generateCARs,
    createCARs,
    top_rules
)
from .data_structures import TransactionDB


class CBA():
    """Class for training a testing the
    CBA Algorithm.

    Parameters:
    -----------
    support : float
    confidence : float
    algorithm : string
        Algorithm for building a classifier.
    maxlen : int
        maximum length of mined rules
    """

    def __init__(self, support=0.10, confidence=0.5, maxlen=10, classification_algorithm="m1", association_algorithm = "apriori"):
        if classification_algorithm not in ["m1", "m3"]:
            raise Exception("algorithm parameter must be either 'm1' or 'm2'!!")
        if 0 > support or support > 1:
            raise Exception("support must be on the interval <0;1>")
        if 0 > confidence or confidence > 1:
            raise Exception("confidence must be on the interval <0;1>")
        if maxlen < 1:
            raise Exception("maxlen cannot be negative or 0")

        self.support = support * 100
        self.confidence = confidence * 100
        self.classification_algorithm = classification_algorithm
        self.association_algorithm = association_algorithm
        self.maxlen = maxlen
        self.clf = None
        self.target_class = None

        self.available_algorithms = {
            "m1": M1Algorithm,
            "m3": M2Algorithm
        }

    def rule_model_accuracy(self, txns):
        """Takes a TransactionDB and outputs
        accuracy of the classifier
        """
        if not self.clf:
            raise Exception("CBA must be trained using fit method first")
        if not isinstance(txns, TransactionDB):
            raise Exception("txns must be of type TransactionDB")

        return self.clf.test_transactions(txns)

    def fit(self, transactions, top_rules_args=0):
        """Trains the model based on input transaction
        and returns self.
        """
        if not isinstance(transactions, TransactionDB):
            raise Exception("transactions must be of type TransactionDB")

        self.target_class = transactions.header[-1]

        used_algorithm = self.available_algorithms[self.classification_algorithm]

        cars = None

        if top_rules_args == 0:
            cars = generateCARs(transactions, assosiation_algorithm=self.association_algorithm, support=self.support, confidence=self.confidence, maxlen=self.maxlen)
        else:
            self.rules = top_rules(transactions.string_representation, assosiation_algorithm=self.association_algorithm,  init_support=self.support, init_conf=self.confidence, appearance=transactions.appeardict, target_rule_count=top_rules_args)
            self.cars = createCARs(self.rules)

        #print(f"rules generated!! rule_len:{len(rules)}, algorithm : {self.association_algorithm}")

        self.algo = used_algorithm(self.cars, transactions)
        self.clf = self.algo.build()

        return self

    def predict(self, X):
        """Method that can be used for predicting
        classes of unseen cases.

        CBA.fit must be used before predicting.
        """
        if not self.clf:
            raise Exception("CBA must be train using fit method first")

        if not isinstance(X, TransactionDB):
            raise Exception("X must be of type TransactionDB")

        return self.clf.predict_all(X)

    def predict_probability(self, X):
        """Method for predicting probablity of
        given classification
¨
        CBA.fit must be used before predicting probablity.
        """

        return self.clf.predict_probability_all(X)

    def predict_matched_rules(self, X):
        """for each data instance, returns a rule that
        matched it according to the CBA order (sorted by
        confidence, support and length)
        """

        return self.clf.predict_matched_rule_all(X)




