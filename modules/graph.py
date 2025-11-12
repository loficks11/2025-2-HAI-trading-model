import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def training_loss(loss_history):
    fig = px.line(
        x=list(range(len(loss_history))),
        y=loss_history,
        labels={"x": "Episode", "y": "Loss"},
        title="Training Loss Over Episodes",
    )
    fig.show()

def action_graph(ticker, prices, actions):
    prices_df = pd.DataFrame(prices, columns=["Index", "Close"])
    actions_df = pd.DataFrame(actions, columns=["Index", "Action"])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=prices_df.index,
            y=prices_df["Close"],
            mode="lines",
            name=f"Prices",
        )
    )

    buy_signals = actions_df[actions_df["Action"] == 1]
    sell_signals = actions_df[actions_df["Action"] == 0]

    fig.add_trace(
        go.Scatter(
            x=prices_df.index[buy_signals["Index"]],
            y=prices_df["Close"].iloc[buy_signals["Index"]],
            mode="markers",
            marker=dict(color="red", symbol="triangle-up", size=10),
            name="Buy Signal",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=prices_df.index[sell_signals["Index"]],
            y=prices_df["Close"].iloc[sell_signals["Index"]],
            mode="markers",
            marker=dict(color="blue", symbol="triangle-down", size=10),
            name="Sell Signal",
        )
    )

    fig.update_layout(
        title=f"Stock Prices with Buy/Sell Signals for {ticker}",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
    )

    fig.show()