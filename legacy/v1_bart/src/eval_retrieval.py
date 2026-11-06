"""
JARVIS-M: Robust Retrieval Benchmark on DialogSum
==================================================
This script evaluates the retrieval module on the ENTIRE DialogSum test set
to scientifically validate retrieval performance with Recall@1, @5, and @10.

Goal: Replace the "100% on 3 queries" claim with rigorous, reproducible metrics.
"""

import os
import torch
import numpy as np
import faiss
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# ============================================================
# Configuration
# ============================================================
DATASET_NAME = "knkarthick/dialogsum"
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
CACHE_DIR = "./cache"

# Retrieval evaluation settings
TOP_K_VALUES = [1, 5, 10]
BATCH_SIZE = 64  # For embedding generation


def setup_device():
    """Setup and return the appropriate device."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("⚠ Using CPU")
    return device


def load_test_dataset():
    """Load the complete DialogSum test set from Hugging Face."""
    print("\n📥 Loading DialogSum dataset...")
    
    # Load the entire dataset to ensure it's downloaded
    dataset = load_dataset(DATASET_NAME, cache_dir=CACHE_DIR)
    
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Validation samples: {len(dataset['validation'])}")
    print(f"  Test samples: {len(dataset['test'])}")
    
    # Use the test split for evaluation
    test_data = dataset["test"]
    
    # Extract dialogues and their corresponding summaries
    dialogues = test_data["dialogue"]
    summaries = test_data["summary"]
    
    print(f"\n📊 Test Set Statistics:")
    print(f"  Number of dialogue-summary pairs: {len(dialogues)}")
    
    # Compute average lengths
    avg_dialogue_len = np.mean([len(d.split()) for d in dialogues])
    avg_summary_len = np.mean([len(s.split()) for s in summaries])
    print(f"  Average dialogue length: {avg_dialogue_len:.1f} words")
    print(f"  Average summary length: {avg_summary_len:.1f} words")
    
    return dialogues, summaries


def build_faiss_index(embeddings):
    """Build a FAISS index for fast similarity search."""
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product after normalization = cosine similarity
    index.add(embeddings)
    
    return index


def compute_recall_at_k(index, query_embeddings, k_values, num_queries):
    """
    Compute Recall@K metrics.
    
    For each query (dialogue embedding), check if the correct summary 
    (at the same index) is retrieved within top-K results.
    """
    max_k = max(k_values)
    
    # Normalize query embeddings
    query_embeddings_normalized = query_embeddings.copy()
    faiss.normalize_L2(query_embeddings_normalized)
    
    # Search for top-k neighbors
    _, indices = index.search(query_embeddings_normalized, max_k)
    
    # Calculate recall at each k
    recalls = {k: 0 for k in k_values}
    
    for query_idx in range(num_queries):
        retrieved_indices = indices[query_idx]
        
        for k in k_values:
            # Check if the correct index (same as query index) is in top-k
            if query_idx in retrieved_indices[:k]:
                recalls[k] += 1
    
    # Convert to percentages
    recall_scores = {k: (count / num_queries) * 100 for k, count in recalls.items()}
    
    return recall_scores, indices


def evaluate_dialogue_to_summary(dialogues, summaries, embedder, device):
    """
    Evaluation Mode 1: Dialogue → Summary retrieval
    Given a dialogue, retrieve the correct summary.
    """
    print("\n" + "=" * 60)
    print("📋 Mode 1: Dialogue → Summary Retrieval")
    print("=" * 60)
    print("Task: Given a dialogue, find its corresponding summary")
    
    # Encode all summaries (these form the search index)
    print("\n🔄 Encoding summaries for index...")
    summary_embeddings = embedder.encode(
        summaries,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device,
    )
    
    # Build FAISS index with summaries
    print("🔧 Building FAISS index...")
    index = build_faiss_index(summary_embeddings.copy())
    
    # Encode all dialogues (these are the queries)
    print("🔄 Encoding dialogues as queries...")
    dialogue_embeddings = embedder.encode(
        dialogues,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device,
    )
    
    # Compute recall metrics
    print("📊 Computing Recall@K metrics...")
    recall_scores, _ = compute_recall_at_k(
        index, dialogue_embeddings, TOP_K_VALUES, len(dialogues)
    )
    
    return recall_scores


def evaluate_summary_to_dialogue(dialogues, summaries, embedder, device):
    """
    Evaluation Mode 2: Summary → Dialogue retrieval
    Given a summary, retrieve the correct dialogue.
    """
    print("\n" + "=" * 60)
    print("📋 Mode 2: Summary → Dialogue Retrieval")
    print("=" * 60)
    print("Task: Given a summary, find its corresponding dialogue")
    
    # Encode all dialogues (these form the search index)
    print("\n🔄 Encoding dialogues for index...")
    dialogue_embeddings = embedder.encode(
        dialogues,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device,
    )
    
    # Build FAISS index with dialogues
    print("🔧 Building FAISS index...")
    index = build_faiss_index(dialogue_embeddings.copy())
    
    # Encode all summaries (these are the queries)
    print("🔄 Encoding summaries as queries...")
    summary_embeddings = embedder.encode(
        summaries,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device,
    )
    
    # Compute recall metrics
    print("📊 Computing Recall@K metrics...")
    recall_scores, _ = compute_recall_at_k(
        index, summary_embeddings, TOP_K_VALUES, len(summaries)
    )
    
    return recall_scores


def evaluate_cross_retrieval(dialogues, summaries, embedder, device):
    """
    Evaluation Mode 3: Cross-modal retrieval (combined index)
    Index contains both dialogues and summaries, check if related pairs cluster.
    """
    print("\n" + "=" * 60)
    print("📋 Mode 3: Cross-Modal Retrieval Analysis")
    print("=" * 60)
    print("Task: Combined analysis with Mean Reciprocal Rank (MRR)")
    
    # Encode both
    print("\n🔄 Encoding all dialogues...")
    dialogue_embeddings = embedder.encode(
        dialogues,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device,
    )
    
    print("🔄 Encoding all summaries...")
    summary_embeddings = embedder.encode(
        summaries,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device,
    )
    
    # Build index with summaries
    summary_emb_copy = summary_embeddings.copy()
    faiss.normalize_L2(summary_emb_copy)
    dim = summary_emb_copy.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(summary_emb_copy)
    
    # Normalize queries
    dialogue_emb_normalized = dialogue_embeddings.copy()
    faiss.normalize_L2(dialogue_emb_normalized)
    
    # Search and compute MRR
    print("📊 Computing Mean Reciprocal Rank (MRR)...")
    max_k = 100  # Search top 100 for MRR calculation
    _, indices = index.search(dialogue_emb_normalized, min(max_k, len(summaries)))
    
    reciprocal_ranks = []
    for query_idx in range(len(dialogues)):
        retrieved_indices = indices[query_idx]
        # Find rank of correct answer
        try:
            rank = np.where(retrieved_indices == query_idx)[0][0] + 1
            reciprocal_ranks.append(1.0 / rank)
        except IndexError:
            reciprocal_ranks.append(0.0)
    
    mrr = np.mean(reciprocal_ranks) * 100
    
    return {"MRR": mrr}


def show_example_retrievals(dialogues, summaries, embedder, device, num_examples=3):
    """Show example successful and failed retrievals for qualitative analysis."""
    print("\n" + "=" * 60)
    print("🔍 Example Retrievals (Qualitative Analysis)")
    print("=" * 60)
    
    # Encode
    summary_embeddings = embedder.encode(
        summaries,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        device=device,
    )
    
    dialogue_embeddings = embedder.encode(
        dialogues,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        device=device,
    )
    
    # Build index
    summary_emb_copy = summary_embeddings.copy()
    faiss.normalize_L2(summary_emb_copy)
    index = faiss.IndexFlatIP(summary_emb_copy.shape[1])
    index.add(summary_emb_copy)
    
    # Normalize queries
    dialogue_emb_normalized = dialogue_embeddings.copy()
    faiss.normalize_L2(dialogue_emb_normalized)
    
    # Search
    distances, indices = index.search(dialogue_emb_normalized, 5)
    
    # Find successes and failures
    successes = []
    failures = []
    
    for i in range(len(dialogues)):
        if indices[i][0] == i:  # Correct retrieval at top-1
            successes.append(i)
        else:
            failures.append(i)
    
    # Show successful examples
    print(f"\n✅ SUCCESSFUL RETRIEVALS ({len(successes)} total)")
    print("-" * 60)
    for idx in successes[:num_examples]:
        print(f"\n[Example {idx}]")
        print(f"Query (dialogue excerpt): {dialogues[idx][:200]}...")
        print(f"Retrieved (top-1 summary): {summaries[indices[idx][0]]}")
        print(f"Correct: ✓")
    
    # Show failed examples
    print(f"\n\n❌ FAILED RETRIEVALS ({len(failures)} total)")
    print("-" * 60)
    for idx in failures[:num_examples]:
        print(f"\n[Example {idx}]")
        print(f"Query (dialogue excerpt): {dialogues[idx][:200]}...")
        print(f"Expected summary: {summaries[idx]}")
        print(f"Retrieved (top-1): {summaries[indices[idx][0]]}")
        
        # Find where correct answer appears
        try:
            correct_rank = np.where(indices[idx] == idx)[0][0] + 1
            print(f"Correct summary rank: {correct_rank}")
        except IndexError:
            print("Correct summary rank: Not in top-5")


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("JARVIS-M: Robust Retrieval Benchmark on DialogSum")
    print("=" * 60)
    
    # Setup
    device = setup_device()
    
    # Load data
    dialogues, summaries = load_test_dataset()
    
    # Load embedder
    print(f"\n📦 Loading SentenceTransformer: {EMBEDDER_MODEL}")
    embedder = SentenceTransformer(EMBEDDER_MODEL, device=device)
    
    # Run evaluations
    results = {}
    
    # Mode 1: Dialogue → Summary
    results["dialogue_to_summary"] = evaluate_dialogue_to_summary(
        dialogues, summaries, embedder, device
    )
    
    # Mode 2: Summary → Dialogue
    results["summary_to_dialogue"] = evaluate_summary_to_dialogue(
        dialogues, summaries, embedder, device
    )
    
    # Mode 3: Cross-modal MRR
    results["cross_modal"] = evaluate_cross_retrieval(
        dialogues, summaries, embedder, device
    )
    
    # Show examples
    show_example_retrievals(dialogues, summaries, embedder, device)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("📈 FINAL EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Test Set Size: {len(dialogues)} dialogue-summary pairs")
    print(f"📦 Embedding Model: {EMBEDDER_MODEL}")
    
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│                   Dialogue → Summary                     │")
    print("├─────────────────────────────────────────────────────────┤")
    for k, score in results["dialogue_to_summary"].items():
        print(f"│  Recall@{k:<2}: {score:>6.2f}%                                  │")
    print("└─────────────────────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│                   Summary → Dialogue                     │")
    print("├─────────────────────────────────────────────────────────┤")
    for k, score in results["summary_to_dialogue"].items():
        print(f"│  Recall@{k:<2}: {score:>6.2f}%                                  │")
    print("└─────────────────────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│                   Cross-Modal Analysis                   │")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│  MRR:      {results['cross_modal']['MRR']:>6.2f}%                                  │")
    print("└─────────────────────────────────────────────────────────┘")
    
    # Summary statistics
    d2s = results["dialogue_to_summary"]
    s2d = results["summary_to_dialogue"]
    
    print("\n📝 Summary:")
    print(f"  • Average Recall@1: {(d2s[1] + s2d[1]) / 2:.2f}%")
    print(f"  • Average Recall@5: {(d2s[5] + s2d[5]) / 2:.2f}%")
    print(f"  • Average Recall@10: {(d2s[10] + s2d[10]) / 2:.2f}%")
    print(f"  • MRR: {results['cross_modal']['MRR']:.2f}%")
    
    print("\n✅ Evaluation complete!")
    print("   These metrics replace the previous '100% on 3 queries' claim")
    print("   with scientifically rigorous, reproducible benchmarks.")
    
    return results


if __name__ == "__main__":
    main()
