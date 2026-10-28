import pandas as pd
import pandera.errors
import pytest

from churn.data import clean_data, download_raw_data, sha256_of, split_train_test, validate


def test_blank_total_charges_become_zero_for_new_customers(raw_df):
    new = raw_df["tenure"] == 0
    assert (raw_df.loc[new, "TotalCharges"].str.strip() == "").all()  # the raw quirk

    clean = clean_data(raw_df)

    assert clean["TotalCharges"].dtype == "float64"
    assert clean["TotalCharges"].notna().all()
    assert (clean.loc[new, "TotalCharges"] == 0.0).all()


def test_other_missing_total_charges_use_tenure_times_monthly(raw_df):
    raw_df.loc[10, "TotalCharges"] = " "
    clean = clean_data(raw_df)
    expected = raw_df.loc[10, "tenure"] * raw_df.loc[10, "MonthlyCharges"]
    assert clean.loc[10, "TotalCharges"] == pytest.approx(expected)


def test_target_is_encoded_as_int(raw_df):
    clean = clean_data(raw_df)
    assert clean["Churn"].dtype == "int64"
    assert set(clean["Churn"].unique()) <= {0, 1}
    assert clean["Churn"].sum() == (raw_df["Churn"] == "Yes").sum()


def test_types_are_coerced(raw_df):
    raw_df["tenure"] = raw_df["tenure"].astype(str)
    raw_df["Contract"] = " " + raw_df["Contract"] + " "  # stray whitespace
    clean = clean_data(raw_df)
    assert clean["tenure"].dtype == "int64"
    assert clean["MonthlyCharges"].dtype == "float64"
    assert set(clean["SeniorCitizen"]) <= {"No", "Yes"}
    assert set(clean["Contract"]) <= {"Month-to-month", "One year", "Two year"}


def test_unknown_target_label_raises(raw_df):
    raw_df.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError, match="Unexpected Churn labels"):
        clean_data(raw_df)


@pytest.mark.parametrize(
    "column, value",
    [
        ("Contract", "Three year"),  # unknown category
        ("tenure", -1),  # out of range
        ("MonthlyCharges", -5.0),  # negative charge
        ("OnlineSecurity", "No internet service"),  # inconsistent with InternetService
    ],
)
def test_schema_rejects_invalid_rows(clean_df, column, value):
    bad = clean_df.copy()
    row = bad.index[bad["InternetService"] != "No"][0]
    bad.loc[row, column] = value
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(bad)


def test_schema_rejects_duplicate_ids(clean_df):
    bad = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(bad)


def test_split_is_stratified_disjoint_and_reproducible(clean_df):
    train, test = split_train_test(clean_df, test_size=0.25, seed=0)
    assert len(train) + len(test) == len(clean_df)
    assert set(train["customerID"]).isdisjoint(test["customerID"])
    assert abs(train["Churn"].mean() - test["Churn"].mean()) < 0.02

    train2, _ = split_train_test(clean_df, test_size=0.25, seed=0)
    pd.testing.assert_frame_equal(train, train2)


def test_download_verifies_checksum(tmp_path):
    src = tmp_path / "source.csv"
    src.write_text("a,b\n1,2\n")
    dest = tmp_path / "raw" / "data.csv"

    path = download_raw_data(src.as_uri(), dest, expected_sha256=sha256_of(src))
    assert path.read_text() == "a,b\n1,2\n"

    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_raw_data(src.as_uri(), dest, expected_sha256="0" * 64, force=True)
