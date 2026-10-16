from noesis.experiments.operating_point import classify


def test_classify_regimes():
    assert classify(0.0, 0.0, 0.001) == "silent"
    assert classify(115.0, 47.0, 0.3) == "saturated"
    assert classify(5.0, 1.0, 0.1) == "sparse"
