import unittest
import numpy as np

from kg_commit.core.window import Window
from kg_commit.model.online_model import OnlineModel
from kg_commit.model.batch_model import BatchModel


class TestOnlineModel(unittest.TestCase):
    def test_online_model_produces_predictions_after_update(self):
        window = Window([
            {"buggy": 0, "feature": 1},
            {"buggy": 1, "feature": 2},
            {"buggy": 0, "feature": 3},
        ], start_time=0, end_time=3)

        window.set_features(np.array([[1.0], [2.0], [3.0]]))
        window.set_labels(np.array([0, 1, 0]))

        model = OnlineModel(classes=[0, 1])
        initial_preds = model.predict(window)
        self.assertEqual(initial_preds.tolist(), [0, 0, 0])

        model.update(window)
        updated_preds = model.predict(window)

        self.assertEqual(updated_preds.shape, (3,))
        self.assertIn(updated_preds[0], [0, 1])
        self.assertIn(updated_preds[1], [0, 1])


class TestBatchModel(unittest.TestCase):
    def test_batch_model_fits_and_predicts(self):
        first = Window([
            {"buggy": 0, "feature": 1},
        ], start_time=0, end_time=0)
        first.set_features(np.array([[1.0]]))
        first.set_labels(np.array([0]))

        second = Window([
            {"buggy": 1, "feature": 2},
        ], start_time=1, end_time=1)
        second.set_features(np.array([[2.0]]))
        second.set_labels(np.array([1]))

        model = BatchModel()
        model.fit([first, second])

        prediction = model.predict(first)
        self.assertEqual(prediction.shape, (1,))
        self.assertIn(prediction[0], [0, 1])
