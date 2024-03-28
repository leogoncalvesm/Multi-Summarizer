from multi_summarizer.processing.utils import custom_cosine

def test_custom_cosine() -> None:
    v1 = [0, 1, 1, 2, 3, 5]
    v2 = [1, 3, 5, 7, 9, 11]
    v3 = [2, 4, 6, 8, 10, 12]
    v4 = [10, 20, 30, 40, 50]

    assert custom_cosine(v1, v2) == 104
    assert custom_cosine(v1, v3) == 116
    assert custom_cosine(v1, v4) == 580
    assert custom_cosine(v2, v3) == 322
    assert custom_cosine(v2, v4) == 1610
    assert custom_cosine(v3, v4) == 1820





