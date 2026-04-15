import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def price_with_anomalies(
    df: pd.DataFrame,
    title: str = "Price with Anomaly Detection",
) -> go.Figure:
    """Candlestick/line chart with anomaly markers overlaid."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=["Price", "Anomaly Score", "Volume"],
    )

    # Price line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["close"],
            mode="lines",
            name="Close Price",
            line=dict(color="#2196F3", width=1.5),
        ),
        row=1,
        col=1,
    )

    # Anomaly markers on price
    if "anomaly" in df.columns:
        anomalies = df[df["anomaly"]]
        fig.add_trace(
            go.Scatter(
                x=anomalies.index,
                y=anomalies["close"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="red", size=8, symbol="circle"),
            ),
            row=1,
            col=1,
        )

    # Ensemble score
    if "ensemble_score" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["ensemble_score"],
                mode="lines",
                name="Ensemble Score",
                line=dict(color="#FF9800", width=1),
                fill="tozeroy",
                fillcolor="rgba(255,152,0,0.1)",
            ),
            row=2,
            col=1,
        )
        # Threshold line
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=2, col=1)

    # Volume bars
    if "volume" in df.columns:
        colors = [
            "red" if a else "#26A69A" for a in df.get("anomaly", [False] * len(df))
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7,
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def model_comparison_bars(comparison_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing model metrics side by side."""
    metrics = ["precision", "recall", "f1"]
    available = [m for m in metrics if m in comparison_df.columns]

    fig = go.Figure()
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    for i, model in enumerate(comparison_df.index):
        fig.add_trace(
            go.Bar(
                name=model,
                x=available,
                y=[comparison_df.loc[model, m] for m in available],
                marker_color=colors[i % len(colors)],
            )
        )

    fig.update_layout(
        title="Model Performance Comparison",
        template="plotly_dark",
        barmode="group",
        yaxis_title="Score",
        height=400,
    )
    return fig


def roc_curves(roc_data: dict[str, dict]) -> go.Figure:
    """Plot ROC curves for multiple models."""
    fig = go.Figure()
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    for i, (name, data) in enumerate(roc_data.items()):
        if not len(data.get("fpr", [])):
            continue
        fig.add_trace(
            go.Scatter(
                x=data["fpr"],
                y=data["tpr"],
                mode="lines",
                name=f'{name} (AUC={data["auc"]:.3f})',
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )

    # Diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="gray", dash="dash"),
        )
    )

    fig.update_layout(
        title="ROC Curves",
        template="plotly_dark",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
    )
    return fig


def anomaly_timeline(
    scores: dict[str, pd.Series],
    threshold: float = 0.5,
) -> go.Figure:
    """Timeline showing anomaly scores from multiple models."""
    fig = go.Figure()
    colors = {"statistical": "#2196F3", "isolation_forest": "#4CAF50", "autoencoder": "#FF9800"}

    for name, series in scores.items():
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=name,
                line=dict(color=colors.get(name, "#E91E63"), width=1),
            )
        )

    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")

    fig.update_layout(
        title="Anomaly Scores Over Time",
        template="plotly_dark",
        yaxis_title="Anomaly Score",
        height=350,
    )
    return fig


def confusion_matrix_heatmap(cm: np.ndarray, model_name: str = "") -> go.Figure:
    """Plot confusion matrix as heatmap."""
    labels = ["Normal", "Anomaly"]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
        )
    )
    fig.update_layout(
        title=f"Confusion Matrix — {model_name}" if model_name else "Confusion Matrix",
        template="plotly_dark",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=350,
        width=400,
    )
    return fig
