import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import wraps

# ---------------------------
# Decorator for logging
# ---------------------------
def log_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[LOG] Running {func.__name__}...")
        result = func(*args, **kwargs)
        print(f"[LOG] Finished {func.__name__}")
        return result
    return wrapper

# ---------------------------
# Economic Data Handler
# ---------------------------
class EconomicData:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    @log_function
    def normalize(self):
        return (self.data - self.data.min()) / (self.data.max() - self.data.min())

# ---------------------------
# Risk Analyzer
# ---------------------------
class RiskAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    @log_function
    def calculate_risk_score(self):
        # Weighted scoring system
        weights = {
            "GDP_growth": -0.4,   # lower growth = higher risk
            "inflation": 0.3,
            "unemployment": 0.2,
            "debt_ratio": 0.5
        }
        score = 0
        for indicator, weight in weights.items():
            if indicator in self.data.columns:
                score += self.data[indicator].mean() * weight
        return score

# ---------------------------
# Forecasting
# ---------------------------
class Forecasting:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    @log_function
    def predict_trend(self, target="GDP_growth"):
        X = np.arange(len(self.data)).reshape(-1, 1)
        y = self.data[target].values
        model = LinearRegression()
        model.fit(X, y)
        future = model.predict([[len(self.data)+1]])
        return future[0]

# ---------------------------
# Policy Document Analyzer
# ---------------------------
class PolicyAnalyzer:
    def __init__(self, documents):
        self.documents = documents

    @log_function
    def extract_topics(self, k=2):
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(self.documents)
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        clusters = {}
        for idx, label in enumerate(kmeans.labels_):
            clusters.setdefault(label, []).append(self.documents[idx])
        return clusters

# ---------------------------
# Main Pipeline
# ---------------------------
class EconomicSecurityProject:
    def __init__(self, data: pd.DataFrame, documents: list):
        self.data = data
        self.documents = documents

    @log_function
    def run(self):
        econ = EconomicData(self.data)
        normalized = econ.normalize()

        risk = RiskAnalyzer(normalized)
        risk_score = risk.calculate_risk_score()

        forecast = Forecasting(normalized)
        gdp_future = forecast.predict_trend("GDP_growth")

        policy = PolicyAnalyzer(self.documents)
        topics = policy.extract_topics(k=2)

        return {
            "risk_score": risk_score,
            "forecast_GDP_growth": gdp_future,
            "policy_topics": topics
        }

# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    # Example economic dataset
    df = pd.DataFrame({
        "GDP_growth": [2.5, 2.0, 1.8, 1.5],
        "inflation": [3.0, 3.5, 4.0, 4.5],
        "unemployment": [5.0, 5.2, 5.5, 6.0],
        "debt_ratio": [60, 65, 70, 75]
    })

    documents = [
        "Government policy focuses on reducing inflation and stabilizing currency.",
        "Trade agreements are being renegotiated to improve economic resilience."
    ]

    project = EconomicSecurityProject(df, documents)
    results = project.run()

    print("\n--- ECONOMIC RISK SCORE ---")
    print(results["risk_score"])

    print("\n--- FORECAST GDP GROWTH ---")
    print(results["forecast_GDP_growth"])

    print("\n--- POLICY TOPICS ---")
    for cluster, docs in results["policy_topics"].items():
        print(f"Cluster {cluster}:")
        for d in docs:
            print(" -", d)
