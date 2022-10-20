"""This is the test for backend API"""


def test_base(test_app):
    """Test response for digit_recognizer base endpoint"""
    response = test_app.get("/digit_recognizer")
    assert response.status_code == 200
    assert response.json() == {"Model Name": "pretrained_resnet50"}


def test_get_env(test_app):
    """Test response for digit_recognizer env endpoint"""
    response = test_app.get("/digit_recognizer/env")
    assert response.status_code == 200
    assert response.json() == {"env": "testing"}
