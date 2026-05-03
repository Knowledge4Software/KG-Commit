class Experiment:

    def __init__(self, dataset, model, preprocessor, stream, evaluator):
        self.dataset = dataset
        self.model = model
        self.preprocessor = preprocessor
        self.stream = stream
        self.evaluator = evaluator

    def run(self):

        while True:

            window = self.stream.next_window()
            if window is None:
                break

            window = self.preprocessor.transform_window(window)

            preds = self.model.predict(window)
            self.model.update(window)

            self.evaluator.add(window, preds)