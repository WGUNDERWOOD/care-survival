class ScoreSplit:
    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test


class Score:
    def __init__(self, ln, lng, rmse, concordance):
        self.ln = ln
        self.lng = lng
        self.rmse = rmse
        self.concordance = concordance
