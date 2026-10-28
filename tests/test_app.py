"""Smoke test of the Streamlit app with Streamlit's AppTest (headless, no browser):
every tab renders without exceptions and reacts to input. Runs offline against a bundle
trained on synthetic data and a synthetic raw CSV."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from conftest import make_raw_telco

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture
def app(monkeypatch, tiny_bundle, tmp_path):
    data_path = tmp_path / "telco.csv"
    make_raw_telco().to_csv(data_path, index=False)
    monkeypatch.setenv("CHURN_MODEL_DIR", str(tiny_bundle))
    monkeypatch.setenv("CHURN_DATA_PATH", str(data_path))
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def metric(at, label):
    return next(m for m in at.metric if m.label == label)


def test_all_tabs_render(app):
    assert not app.exception
    assert not app.error
    assert [t.label for t in app.tabs] == [
        "Customer what-if",
        "Portfolio targeting",
        "Survival by segment",
    ]
    assert metric(app, "Churn probability").value.endswith("%")
    assert metric(app, "Customers targeted").value


def test_what_if_reacts_and_compares_to_a_pinned_baseline(app):
    before = metric(app, "Churn probability").value
    app.button[0].click().run()  # pin the default customer
    app.selectbox(key="in_Contract").set_value("Two year").run()
    app.slider(key="in_tenure").set_value(66).run()

    assert not app.exception
    assert metric(app, "Churn probability").value != before
    assert metric(app, "Churn probability").delta  # change vs the pinned baseline
    assert metric(app, "12-month churn").value == "n/a"  # 66 + 12 > 72 months observed


def test_portfolio_threshold_changes_the_campaign(app):
    targeted = int(metric(app, "Customers targeted").value.replace(",", ""))
    app.slider[1].set_value(0.95).run()  # the tenure slider comes first
    assert not app.exception
    assert int(metric(app, "Customers targeted").value.replace(",", "")) < targeted


def test_segment_views(app):
    app.selectbox[-1].set_value("PaymentMethod").run()
    app.button_group[0].set_value("model").run()
    assert not app.exception
