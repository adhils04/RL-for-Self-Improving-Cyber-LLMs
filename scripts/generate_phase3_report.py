import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

# Hardcoded paths based on the extracted Kaggle ZIP
MAPPO_BASELINE_JSON = REPORTS_DIR / "mappo_coevolution_review2_coevolution_metrics.json"
INDEPENDENT_PPO_JSON = REPORTS_DIR / "review2_coevolution_metrics.json"

def smooth_series(series, window_size=20):
    """Applies a moving average to smooth the plot."""
    return series.rolling(window=window_size, min_periods=1).mean()

def load_data():
    if not MAPPO_BASELINE_JSON.exists():
        print(f"Error: Could not find {MAPPO_BASELINE_JSON}")
        sys.exit(1)
        
    if not INDEPENDENT_PPO_JSON.exists():
        print(f"Error: Could not find {INDEPENDENT_PPO_JSON}")
        sys.exit(1)
        
    with open(MAPPO_BASELINE_JSON, 'r') as f:
        mappo_data = json.load(f)
        
    with open(INDEPENDENT_PPO_JSON, 'r') as f:
        indep_data = json.load(f)
        
    df_mappo = pd.DataFrame(mappo_data["asr_vs_step"])
    df_indep = pd.DataFrame(indep_data["asr_vs_step"])
    
    return df_mappo, df_indep

def plot_asr(df_mappo, df_indep):
    plt.figure(figsize=(10, 6))
    
    # Smooth ASR (Attack Success Rate)
    mappo_asr = smooth_series(df_mappo['asr'], 50)
    indep_asr = smooth_series(df_indep['asr'], 50)
    
    plt.plot(df_mappo['global_step'], mappo_asr, label='MAPPO (Centralized Critic)', color='#1f77b4', linewidth=2)
    plt.plot(df_indep['global_step'], indep_asr, label='Independent PPO (Ablation)', color='#ff7f0e', linewidth=2, linestyle='--')
    
    plt.title('Attack Success Rate (ASR) Over Time\nMAPPO vs. Independent PPO', fontsize=14, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Attack Success Rate (Smoothed)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    out_path = REPORTS_DIR / 'asr_comparison.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()

def plot_rewards(df_mappo, df_indep):
    plt.figure(figsize=(10, 6))
    
    # Smooth Attacker Rewards
    mappo_rew = smooth_series(df_mappo['attacker_reward'], 50)
    indep_rew = smooth_series(df_indep['attacker_reward'], 50)
    
    plt.plot(df_mappo['global_step'], mappo_rew, label='MAPPO Attacker Reward', color='#2ca02c', linewidth=2)
    plt.plot(df_indep['global_step'], indep_rew, label='Independent PPO Attacker Reward', color='#d62728', linewidth=2, linestyle='--')
    
    plt.title('Attacker Rewards Over Time\nCentralized vs. Decentralized Critic', fontsize=14, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Attacker Reward (Smoothed)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = REPORTS_DIR / 'reward_comparison.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()

def plot_critic_loss(df_mappo, df_indep):
    # Only if critic_loss exists in the JSON, but our wandb logger might not have saved it in asr_vs_step!
    # Let's check if it exists in the DataFrame
    if 'critic_loss' in df_mappo.columns and 'critic_loss' in df_indep.columns:
        plt.figure(figsize=(10, 6))
        
        mappo_loss = smooth_series(df_mappo['critic_loss'], 50)
        indep_loss = smooth_series(df_indep['critic_loss'], 50)
        
        plt.plot(df_mappo['global_step'], mappo_loss, label='MAPPO Critic Loss', color='#9467bd', linewidth=2)
        plt.plot(df_indep['global_step'], indep_loss, label='Independent PPO Critic Loss', color='#8c564b', linewidth=2, linestyle='--')
        
        plt.title('Critic Loss Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss (Smoothed)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        out_path = REPORTS_DIR / 'critic_loss_comparison.png'
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        print(f"Saved {out_path}")
        plt.close()

def main():
    print("Loading Kaggle JSON metrics...")
    df_mappo, df_indep = load_data()
    
    print(f"Found {len(df_mappo)} steps for MAPPO Baseline.")
    print(f"Found {len(df_indep)} steps for Independent PPO.")
    
    plot_asr(df_mappo, df_indep)
    plot_rewards(df_mappo, df_indep)
    plot_critic_loss(df_mappo, df_indep)
    
    print("All Phase 3 charts generated successfully!")

if __name__ == "__main__":
    main()
