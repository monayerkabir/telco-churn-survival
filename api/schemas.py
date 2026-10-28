"""Request and response models for the churn API.

Field names follow the IBM Telco dataset so the same record format works for training
data, the API and the app.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from churn.config import ADDON_SERVICE_COLS, MAX_TENURE

MAX_BATCH_SIZE = 1000

YesNo = Literal["Yes", "No"]
AddOn = Literal["Yes", "No", "No internet service"]

EXAMPLE_CUSTOMER = {
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.5,
    "TotalCharges": 447.5,
}


class CustomerFeatures(BaseModel):
    """One customer's attributes at scoring time."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_CUSTOMER]})

    customerID: str | None = Field(
        None, max_length=64, description="Optional; echoed back, not used by the model."
    )
    gender: Literal["Female", "Male"]
    SeniorCitizen: YesNo = Field(description='"Yes"/"No"; 1/0 and true/false are accepted.')
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(
        ge=0, le=MAX_TENURE, description="Months with the company (0-72, the range trained on)."
    )
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: AddOn
    OnlineBackup: AddOn
    DeviceProtection: AddOn
    TechSupport: AddOn
    StreamingTV: AddOn
    StreamingMovies: AddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(gt=0, le=500, description="Current monthly bill (USD).")
    TotalCharges: float | None = Field(
        None, ge=0, description="Billed to date (USD). Defaults to tenure × MonthlyCharges."
    )

    @field_validator("SeniorCitizen", mode="before")
    @classmethod
    def _senior_from_flag(cls, value):
        if isinstance(value, bool | int):  # the raw dataset encodes this flag as 0/1
            return "Yes" if value else "No"
        return value

    @model_validator(mode="after")
    def _services_are_consistent(self) -> CustomerFeatures:
        """Reject contradictions; let clients send plain "No" for services they cannot have."""
        no_internet = self.InternetService == "No"
        for col in ADDON_SERVICE_COLS:
            value = getattr(self, col)
            if no_internet and value == "Yes":
                raise ValueError(f"{col} cannot be 'Yes' when InternetService is 'No'")
            if not no_internet and value == "No internet service":
                raise ValueError(
                    f"{col} is 'No internet service' but InternetService is '{self.InternetService}'"
                )
            if no_internet:
                object.__setattr__(self, col, "No internet service")

        if self.PhoneService == "No" and self.MultipleLines == "Yes":
            raise ValueError("MultipleLines cannot be 'Yes' when PhoneService is 'No'")
        if self.PhoneService == "Yes" and self.MultipleLines == "No phone service":
            raise ValueError("MultipleLines is 'No phone service' but PhoneService is 'Yes'")
        if self.PhoneService == "No":
            object.__setattr__(self, "MultipleLines", "No phone service")
        return self

    def features(self) -> dict:
        """The model inputs (everything except the ID)."""
        return self.model_dump(exclude={"customerID"})


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


# --------------------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------------------
class RevenueAtRisk(BaseModel):
    horizon_months: int
    amount: float = Field(description="P(churn) × MonthlyCharges × horizon (USD).")


class Decision(BaseModel):
    target: bool = Field(description="P(churn) ≥ the retention threshold chosen in training.")
    threshold: float
    value_aware_target: bool = Field(
        description="Targeting this customer has positive expected profit on its own."
    )
    breakeven_probability: float
    expected_profit_if_targeted: float = Field(
        description="P(churn) × success rate × value − offer cost (USD)."
    )


class HorizonSurvival(BaseModel):
    months_ahead: int
    probability: float | None = Field(
        description="P(still a customer | active today); null beyond the 72-month window."
    )


class SurvivalPoint(BaseModel):
    month: int
    survival: float


class SurvivalSummary(BaseModel):
    model: str
    current_tenure_months: int
    survival_probability: list[HorizonSurvival]
    churn_probability_12m: float | None
    expected_active_months_next_12m: float | None = Field(
        description="E[min(T − t₀, 12) | T > t₀]: comparable across customers."
    )
    expected_revenue_lost_next_12m: float | None
    expected_remaining_tenure_months: float = Field(
        description="E[min(T, 72) − t₀ | T > t₀]: restricted to the 72-month observation window."
    )
    observation_window_months: int
    curve: list[SurvivalPoint] | None = Field(
        description="Conditional survival from today's tenure; only with include_curve=true."
    )


class Driver(BaseModel):
    feature: str
    value: str | float
    contribution: float = Field(description="SHAP value in log-odds of churn.")
    direction: Literal["increases_risk", "decreases_risk"]


class Explanation(BaseModel):
    base_log_odds: float
    top_drivers: list[Driver]
    other_features_contribution: float


class Prediction(BaseModel):
    customerID: str | None
    churn_probability: float = Field(ge=0, le=1)
    revenue_at_risk: RevenueAtRisk
    decision: Decision
    survival: SurvivalSummary
    explanation: Explanation


class PredictionResponse(Prediction):
    model_version: str


class BatchResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    champion: str
    survival_model: str
    threshold: float
