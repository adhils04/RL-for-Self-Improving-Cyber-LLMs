"""
scripts/analyze_attack_diversity.py

Loads adversarial traces, computes embeddings using sentence-transformers, 
and calculates average pairwise cosine distance to measure attack diversity.
"""
import json
import numpy as np
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_distances
except ImportError:
    print("Please install required packages: pip install sentence-transformers scikit-learn")
    exit(1)

def main():
    report_path = Path("reports/review2_adversarial_traces.json")
    if not report_path.exists():
        print(f"File not found: {report_path}")
        return

    with open(report_path, "r") as f:
        data = json.load(f)

    payloads = []
    
    # Check successful attacks
    for case in data.get("successful_attacks", []):
        payload = case.get("attacker_payload") or case.get("defender_answer_snippet", "")
        if payload:
            payloads.append(payload)

    # Check defended cases
    for case in data.get("defended_cases", []):
        payload = case.get("attacker_payload") or case.get("defender_answer_snippet", "")
        if payload:
            payloads.append(payload)

    if not payloads:
        print("No payloads found to analyze.")
        return

    print(f"Loaded {len(payloads)} payloads. Computing embeddings using 'all-MiniLM-L6-v2'...")

    # Use a lightweight CPU-friendly model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(payloads)

    # Compute pairwise cosine distances
    distances = cosine_distances(embeddings)
    
    # Calculate the average distance for the upper triangle (excluding identical pairs on the diagonal)
    n = len(payloads)
    if n > 1:
        upper_tri_indices = np.triu_indices(n, k=1)
        avg_distance = np.mean(distances[upper_tri_indices])
    else:
        avg_distance = 0.0

    print(f"\n--- Attack Diversity Analysis ---")
    print(f"Total Traces Analyzed: {n}")
    print(f"Average Pairwise Cosine Distance: {avg_distance:.4f}")
    
    if avg_distance > 0.5:
        print("Conclusion: HIGH diversity. The Attacker explored varied prompt patterns.")
    elif avg_distance > 0.2:
        print("Conclusion: MODERATE diversity.")
    else:
        print("Conclusion: LOW diversity. The Attacker may have collapsed to a single strategy.")

if __name__ == "__main__":
    main()
