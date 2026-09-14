"""
tests/test_api.py — smoke tests for predict + API

Run:
  py -3.13 -m pytest tests/test_api.py -v
  py -3.13 tests/test_api.py
"""
import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT))

from fastapi.testclient import TestClient
from simulation.backend.api import app


def test_health():
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    # model may or may not be loaded in CI without artifacts
    assert "model_loaded" in j


def test_stats():
    c = TestClient(app)
    r = c.get("/stats")
    assert r.status_code == 200
    j = r.json()
    assert "rows" in j
    assert j["rows"] == 640 or j["rows"] > 600


def test_predict_basic():
    c = TestClient(app)
    payload = {
        "Year": 2000, "MAKE": "TOYOTA", "MODEL": "COROLLA",
        "VEHICLE_CLASS": "COMPACT", "ENGINE_SIZE": 2.0,
        "CYLINDERS": 4, "TRANSMISSION": "A4", "FUEL": "X",
        "FUEL_CONSUMPTION": 10.5
    }
    r = c.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    j = r.json()
    for k in ["co2", "fuel_consumption", "physics_base", "co2_residual", "model"]:
        assert k in j
    # sanity: CO2 around 200-300 for 10.5 FC
    assert 120 < j["co2"] < 400
    # physics base must be ~ 248.85
    assert abs(j["physics_base"] - 10.5*23.7) < 1e-3


def test_predict_with_speed():
    c = TestClient(app)
    payload = {
        "Year": 2000, "MAKE": "TOYOTA", "MODEL": "COROLLA",
        "VEHICLE_CLASS": "COMPACT", "ENGINE_SIZE": 2.0,
        "CYLINDERS": 4, "TRANSMISSION": "A4", "FUEL": "X",
        "FUEL_CONSUMPTION": 10.5, "speed": 120
    }
    r = c.post("/predict", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert j.get("speed_factor") is not None
    assert j["speed_factor"] > 1.0
    assert j["co2"] > 200  # amplified


def test_predict_batch():
    c = TestClient(app)
    batch = [
        {"ENGINE_SIZE": 1.6, "CYLINDERS": 4, "FUEL_CONSUMPTION": 9.0, "VEHICLE_CLASS": "COMPACT", "TRANSMISSION": "A4", "FUEL": "X", "MAKE": "HONDA", "MODEL": "CIVIC", "Year": 2000},
        {"ENGINE_SIZE": 5.0, "CYLINDERS": 8, "FUEL_CONSUMPTION": 18.0, "VEHICLE_CLASS": "SUV", "TRANSMISSION": "A4", "FUEL": "X", "MAKE": "FORD", "MODEL": "F150", "Year": 2000},
    ]
    r = c.post("/predict_batch", json=batch)
    assert r.status_code == 200
    arr = r.json()
    assert len(arr) == 2
    assert arr[1]["co2"] > arr[0]["co2"]  # SUV larger should generally be higher


def test_predict_one_direct():
    from src.predict import predict_one
    res = predict_one({
        "ENGINE_SIZE": 2.0, "CYLINDERS": 4, "FUEL_CONSUMPTION": 10.5, "FC": 10.5,
        "VEHICLE_CLASS": "COMPACT", "TRANSMISSION": "A4", "FUEL": "X",
        "MAKE": "TOYOTA", "MODEL": "COROLLA", "Year": 2000
    })
    assert "co2" in res
    assert res["model"] in ("fuelnet", "fallback")


if __name__ == "__main__":
    test_health(); print("health OK")
    test_stats(); print("stats OK")
    test_predict_basic(); print("predict OK")
    test_predict_with_speed(); print("predict+speed OK")
    test_predict_batch(); print("batch OK")
    test_predict_one_direct(); print("direct OK")
    print("All smoke tests passed.")

