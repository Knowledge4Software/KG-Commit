from kg_commit.core.window import Window


def test_window_preprocess_state_and_repr():
    commits = [{"buggy": 0}, {"buggy": 1}]
    window = Window(commits, start_time=0, end_time=1)

    assert not window.is_preprocessed()
    window.set_features([[1.0], [2.0]])
    window.set_labels([0, 1])

    assert window.is_preprocessed()
    assert window.get_features()[1][0] == 2.0
    assert window.get_labels().tolist() == [0, 1]
    assert "Window(size=2" in repr(window)
