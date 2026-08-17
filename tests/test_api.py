def test_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rejects_bad_file_type(client):
    response = client.post(
        "/api/v1/predict",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
