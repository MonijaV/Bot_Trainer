import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from modules.nlu_pipeline import predict
from modules.data_loader import load_eval_dataset, load_intents, get_intent_names

def run_evaluation():
    """
    Runs the full NLU pipeline on all eval samples,
    computes metrics, saves results and confusion matrix chart.
    """
    print("=" * 60)
    print("BOTTRAINER — NLU EVALUATION")
    print("=" * 60)

    # Load data
    eval_samples = load_eval_dataset()
    intents_data = load_intents()
    intent_names = get_intent_names(intents_data)

    print(f"\nTotal test samples : {len(eval_samples)}")
    print(f"Total intent classes: {len(intent_names)}")
    print("\nRunning predictions...\n")

    y_true = []   # expected intents
    y_pred = []   # predicted intents
    rows   = []   # for the results table

    for i, sample in enumerate(eval_samples):
        result     = predict(sample["text"])
        expected   = sample["expected_intent"]
        predicted  = result["intent"]
        confidence = result["confidence"]
        is_correct = expected == predicted

        y_true.append(expected)
        y_pred.append(predicted)

        rows.append({
            "id":         sample["id"],
            "text":       sample["text"],
            "expected":   expected,
            "predicted":  predicted,
            "confidence": round(confidence, 2),
            "correct":    "✅" if is_correct else "❌"
        })

        status = "✅" if is_correct else "❌"
        print(f"[{i+1:02d}] {status}  {sample['text'][:45]:<45} "
              f"→ {predicted:<20} (conf: {confidence:.2f})")

    print("\n" + "=" * 60)

    # ── Core Metrics ──────────────────────────────────────────
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted",
                                zero_division=0)
    recall    = recall_score(y_true, y_pred, average="weighted",
                             zero_division=0)
    f1        = f1_score(y_true, y_pred, average="weighted",
                         zero_division=0)

    print(f"\n{'METRIC':<20} {'SCORE'}")
    print("-" * 35)
    print(f"{'Accuracy':<20} {accuracy*100:.1f}%")
    print(f"{'Precision (weighted)':<20} {precision*100:.1f}%")
    print(f"{'Recall (weighted)':<20} {recall*100:.1f}%")
    print(f"{'F1 Score (weighted)':<20} {f1*100:.1f}%")
    print("-" * 35)

    # ── Per-Intent Report ─────────────────────────────────────
    print("\nPER-INTENT CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred,
                                 labels=intent_names,
                                 zero_division=0))

    # ── Save Results CSV ──────────────────────────────────────
    os.makedirs("evaluation/results", exist_ok=True)

    df = pd.DataFrame(rows)
    csv_path = "evaluation/results/eval_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved → {csv_path}")

    # ── Save Metrics JSON ─────────────────────────────────────
    metrics = {
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1_score":  round(f1, 4),
        "total_samples": len(eval_samples),
        "correct":   int(accuracy * len(eval_samples)),
        "wrong":     len(eval_samples) - int(accuracy * len(eval_samples))
    }
    metrics_path = "evaluation/results/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved  → {metrics_path}")

    # ── Confusion Matrix Chart ────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=intent_names)

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=intent_names,
        yticklabels=intent_names,
        linewidths=0.5,
        linecolor="#1a1a2e",
        cbar_kws={"shrink": 0.8}
    )
    plt.title("BotTrainer — Confusion Matrix", fontsize=15,
              fontweight="bold", pad=16)
    plt.ylabel("Actual Intent",    fontsize=11)
    plt.xlabel("Predicted Intent", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()

    chart_path = "evaluation/results/confusion_matrix.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix → {chart_path}")

    # ── F1 Per Intent Bar Chart ───────────────────────────────
    report = classification_report(y_true, y_pred,
                                   labels=intent_names,
                                   output_dict=True,
                                   zero_division=0)
    f1_scores = [report[i]["f1-score"] for i in intent_names]

    plt.figure(figsize=(12, 5))
    colors = ["#10B981" if s >= 0.9 else "#F59E0B"
              if s >= 0.7 else "#EF4444" for s in f1_scores]
    bars = plt.bar(intent_names, f1_scores, color=colors,
                   edgecolor="none", width=0.6)

    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{score:.2f}",
                 ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    plt.title("F1 Score per Intent", fontsize=14,
              fontweight="bold", pad=14)
    plt.xlabel("Intent",   fontsize=11)
    plt.ylabel("F1 Score", fontsize=11)
    plt.ylim(0, 1.12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.axhline(y=0.9, color="#10B981", linestyle="--",
                linewidth=1, alpha=0.5, label="90% threshold")
    plt.legend(fontsize=9)
    plt.tight_layout()

    f1_path = "evaluation/results/f1_per_intent.png"
    plt.savefig(f1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"F1 chart saved → {f1_path}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    run_evaluation()