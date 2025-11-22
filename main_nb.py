import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    sns.set_theme(style="whitegrid")
    return np, pd, plt, sns, stats


@app.cell
def _(np, pd):
    n_A = 2500
    n_B = 2500
    p_A = 0.040
    p_B = 0.058

    group_A_data = np.random.choice([0, 1], size=n_A, p=[1-p_A, p_A])
    group_B_data = np.random.choice([0, 1], size=n_B, p=[1-p_B, p_B])

    df_ab = pd.DataFrame({
        'user_id': range(1, n_A + n_B + 1),
        'group': ['A'] * n_A + ['B'] * n_B,
        'converted': np.concatenate([group_A_data, group_B_data])
    })

    df_ab = df_ab.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total Users: {len(df_ab)}")
    df_ab.head()
    return (df_ab,)


@app.cell
def _(df_ab):
    ab_summary = df_ab.groupby('group')['converted'].agg(['count', 'sum', 'mean']).reset_index()
    ab_summary.columns = ['Group', 'Total Users', 'Converted Users', 'Conversion Rate']

    print(ab_summary)
    return (ab_summary,)


@app.cell
def _(ab_summary, plt, sns):
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Group', y='Conversion Rate', hue='Group', data=ab_summary, palette=['gray', 'green'], legend=False)
    plt.title('Conversion Rate: Group A vs Group B')
    plt.ylim(0, 0.1)
    plt.ylabel('Conversion Rate (Proportion)')
    plt.show()
    return


@app.cell
def _(df_ab, pd, stats):
    contingency_table = pd.crosstab(df_ab['group'], df_ab['converted'])

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    print("Contingency Table:")
    print(contingency_table)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.4f}")

    alpha = 0.05
    if p_value < alpha:
        print("\nResult: Statistically Significant! (Reject H0)")
    else:
        print("\nResult: Not Significant (Fail to Reject H0)")
    return


if __name__ == "__main__":
    app.run()
