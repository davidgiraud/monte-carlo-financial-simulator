
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.set_page_config(page_title="Monte Carlo Financial Simulator", layout="wide")

st.title("Monte Carlo Financial Simulator")
st.markdown("Simulate your investment growth over time with adjustable parameters and visualize uncertainty.")

def monte_carlo_financial_model_ultimate(
    initial_investment,
    base_annual_contribution,
    years,
    inflation_rate,
    mean_return,
    std_dev,
    simulations,
    annual_contribution_growth,
    contribution_timing,
    random_seed=42
):
    np.random.seed(random_seed)
    all_simulations = np.zeros((simulations, years))

    for sim in range(simulations):
        balance = initial_investment
        contribution = base_annual_contribution

        for year in range(years):
            random_return = np.random.normal(mean_return, std_dev)
            if contribution_timing == 'beginning':
                balance += contribution
                balance *= (1 + random_return / 100)
            else:
                balance *= (1 + random_return / 100)
                balance += contribution
            real_balance = balance / ((1 + inflation_rate / 100) ** (year + 1))
            all_simulations[sim, year] = real_balance
            contribution *= (1 + annual_contribution_growth / 100)

    return all_simulations

st.sidebar.header("Simulation Settings")

initial_investment = st.sidebar.number_input("Initial Investment ($)", min_value=0, value=10000, step=1000)
base_annual_contribution = st.sidebar.number_input("Annual Contribution ($)", min_value=0, value=5000, step=500)
annual_contribution_growth = st.sidebar.slider("Contribution Growth Rate (%)", 0.0, 10.0, 3.0, step=0.1)
years = st.sidebar.slider("Number of Years", 5, 50, 30)
mean_return = st.sidebar.slider("Expected Annual Return (%)", 0.0, 15.0, 7.0, step=0.1)
std_dev = st.sidebar.slider("Return Volatility (Std Dev %)", 0.0, 15.0, 5.0, step=0.1)
inflation_rate = st.sidebar.slider("Annual Inflation Rate (%)", 0.0, 10.0, 2.5, step=0.1)
contribution_timing = st.sidebar.selectbox("Contribution Timing", ['beginning', 'end'])
simulations = st.sidebar.number_input("Number of Simulations", min_value=100, value=1000, step=100)

results = monte_carlo_financial_model_ultimate(
    initial_investment,
    base_annual_contribution,
    years,
    inflation_rate,
    mean_return,
    std_dev,
    simulations,
    annual_contribution_growth,
    contribution_timing
)

median_projection = np.median(results, axis=0)
percentile_10 = np.percentile(results, 10, axis=0)
percentile_90 = np.percentile(results, 90, axis=0)
final_balances = results[:, -1]
years_range = np.arange(1, years + 1)
target_goals = [500000, 750000, 1000000]
goal_probabilities = {goal: np.mean(final_balances >= goal) * 100 for goal in target_goals}

st.subheader("Investment Growth Over Time")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(years_range, median_projection, label="Median Projection", color='blue')
ax.fill_between(years_range, percentile_10, percentile_90, color='lightgray', alpha=0.5, label="80% Confidence Interval")
ax.set_xlabel("Year")
ax.set_ylabel("Real Portfolio Value ($)")
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()
st.pyplot(fig)

st.subheader("Distribution of Final Real Portfolio Values")
fig2, ax2 = plt.subplots(figsize=(12,6))
sns.histplot(final_balances, bins=40, kde=True, color='skyblue', ax=ax2)
for goal in target_goals:
    ax2.axvline(goal, linestyle='dashed', linewidth=2, label=f"Target: ${goal:,}")
ax2.set_xlabel("Final Real Portfolio Value ($)")
ax2.set_ylabel("Frequency")
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()
st.pyplot(fig2)

st.subheader("Goal Achievement Probabilities")
goal_summary = pd.DataFrame({
    "Target Amount ($)": target_goals,
    "Probability of Success (%)": [goal_probabilities[goal] for goal in target_goals]
})
st.dataframe(goal_summary)
