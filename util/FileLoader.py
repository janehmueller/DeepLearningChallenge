import json


class File:
    def __init__(self, path):
        self.path = path
        self.data = None
        self.preprocess()

    def preprocess(self):
        with open(self.path) as file:
            self.data = json.load(file)
