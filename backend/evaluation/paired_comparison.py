import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def run_analysis():
    results_path = Path(__file__).parent / 'results' / 'repeated_seed_policy_results.csv'
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        return

    df = pd.read_csv(results_path)
    
    # Pivot the data to have seeds as index and policies as columns
    pivot = df.pivot(index='seed', columns='policy', values='policy_value')
    
    comparisons = [
        ('causapay', 'naive_likelihood_matched_volume', 'CausaPay vs Naive'),
        ('causapay', 'consent_safe_baseline', 'CausaPay vs Baseline'),
    ]
    
    print("=== Paired Policy Comparison Analysis ===\n")
    
    for p1, p2, label in comparisons:
        if p1 not in pivot.columns or p2 not in pivot.columns:
            print(f"Skipping {label}: one or both policies not found in results.\n")
            continue
            
        delta = pivot[p1] - pivot[p2]
        
        # Basic statistics
        mean_diff = delta.mean()
        median_diff = delta.median()
        std_diff = delta.std()
        
        # Win rate (sign test)
        wins = (delta > 0).sum()
        losses = (delta < 0).sum()
        ties = (delta == 0).sum()
        win_rate = wins / len(delta)
        
        # Simple paired t-test
        t_stat, p_val = stats.ttest_rel(pivot[p1], pivot[p2])
        
        # Sign test (non-parametric)
        sign_p_val = stats.binomtest(wins, n=wins+losses, p=0.5, alternative='greater').pvalue if (wins+losses) > 0 else np.nan
        
        print(f"--- {label} ---")
        print(f"Mean Delta:   INR {mean_diff:,.2f}")
        print(f"Median Delta: INR {median_diff:,.2f}")
        print(f"Std Dev:      INR {std_diff:,.2f}")
        print(f"Wins/Losses/Ties: {wins}/{losses}/{ties}")
        print(f"Win Rate:     {win_rate*100:.1f}%")
        print(f"t-test p-val: {p_val:.4f}")
        print(f"Sign test p-val (wins > losses): {sign_p_val:.4f}")
        print("\n")

if __name__ == '__main__':
    run_analysis()
