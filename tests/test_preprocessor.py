import unittest

from kg_commit.core.window import Window
from kg_commit.data.preprocess import SimplePreprocessor


class TestSimplePreprocessor(unittest.TestCase):
    def test_transform_window_creates_features_and_labels(self):
        commits = [
            {"buggy": 0, "files_changed": 3, "message": "fix bug"},
            {"buggy": 1, "files_changed": 5, "message": "add feature"},
        ]
        window = Window(commits, start_time=0, end_time=1)
        preprocessor = SimplePreprocessor(label_column="buggy")

        preprocessor.transform_window(window)

        self.assertTrue(window.is_preprocessed())
        self.assertEqual(window.get_features().shape, (2, 1))
        self.assertEqual(window.get_labels().tolist(), [0, 1])
