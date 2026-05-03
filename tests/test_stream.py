import unittest

from kg_commit.data.stream import CommitStream


class TestCommitStream(unittest.TestCase):
    def test_next_window_chunks_commits(self):
        commits = [{"commit_time": i, "buggy": i % 2} for i in range(10)]
        stream = CommitStream(commits, window_size=4, step=4)

        window = stream.next_window()
        self.assertEqual(window.size(), 4)
        self.assertEqual(window.start_time, 0)
        self.assertEqual(window.end_time, 3)

        window = stream.next_window()
        self.assertEqual(window.size(), 4)
        self.assertEqual(window.start_time, 4)
        self.assertEqual(window.end_time, 7)

        window = stream.next_window()
        self.assertEqual(window.size(), 2)
        self.assertEqual(window.start_time, 8)
        self.assertEqual(window.end_time, 9)

        self.assertIsNone(stream.next_window())
