import json
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from pathlib import Path

# Path to the global results JSON file
global_results_path = "evaluation_results/global_results.json"

# Directory to save the results
output_dir = Path("tmp")
output_dir.mkdir(exist_ok=True, parents=True)

# Load all results
with open(global_results_path, "r") as f:
    global_results = json.load(f)

# global_results is a list of dicts, one per sample_index
# Each entry has keys like:
# "sample_index", "mIoU_vs_n_masks", "correlations", "mean_mIoU", "num_objects", "hard_labels"

# --- Aggregate results across the entire dataset ---

# 1. Aggregate mIoU for each number of masks
all_n_masks = set()
for res in global_results:
    for n_mask in res["mIoU_vs_n_masks"].keys():
        all_n_masks.add(int(n_mask))
all_n_masks = sorted(list(all_n_masks))

mious_per_n = {n: [] for n in all_n_masks}
for res in global_results:
    for n_mask_str, val in res["mIoU_vs_n_masks"].items():
        n_mask = int(n_mask_str)
        mious_per_n[n_mask].append(val)

avg_mious_per_n = {n: mean(vals) if len(vals) > 0 else 0.0 for n, vals in mious_per_n.items()}

# 2. Aggregate correlations
attributes = []
for res in global_results:
    for attr in res["correlations"].keys():
        if attr not in attributes:
            attributes.append(attr)

correlations_attr = {attr: [] for attr in attributes}
for res in global_results:
    for attr in attributes:
        if attr in res["correlations"]:
            correlations_attr[attr].append(res["correlations"][attr])

avg_correlations = {attr: mean(vals) if len(vals) > 0 else 0.0 for attr, vals in correlations_attr.items()}

# 3. Distribution of the number of objects
num_objects_list = [res["num_objects"] for res in global_results]

# 4. Identify best and worst performing images (e.g., by mean_mIoU)
mean_mious = [(res["sample_index"], res["mean_mIoU"]) for res in global_results if "mean_mIoU" in res]
mean_mious.sort(key=lambda x: x[1])  # sort by mean_mIoU
worst_performing = mean_mious[:5]  # bottom 5
best_performing = mean_mious[-5:]   # top 5

# --- Print insights and save as text file ---

summary_file = output_dir / "summary.txt"
with summary_file.open("w") as f:
    f.write("=== Average mIoU for each number of masks (across all images) ===\n")
    for n in avg_mious_per_n:
        f.write(f"{n} masks: {avg_mious_per_n[n]:.4f}\n")
    
    f.write("\n=== Average correlations with region properties (Spearman) ===\n")
    for attr, val in avg_correlations.items():
        f.write(f"{attr}: {val:.4f}\n")
    
    f.write("\n=== Distribution of number of objects per image ===\n")
    f.write(f"Min objects: {min(num_objects_list)}\n")
    f.write(f"Max objects: {max(num_objects_list)}\n")
    f.write(f"Mean objects: {mean(num_objects_list):.2f}\n")
    
    f.write("\n=== Best performing images (by mean_mIoU) ===\n")
    for idx, m in best_performing:
        f.write(f"Image {idx}: mean_mIoU={m:.4f}\n")
    
    f.write("\n=== Worst performing images (by mean_mIoU) ===\n")
    for idx, m in worst_performing:
        f.write(f"Image {idx}: mean_mIoU={m:.4f}\n")

print(f"Summary saved to {summary_file}")

# --- Plot results and save to the output directory ---

# Plot average mIoU vs number of masks
plt.figure(figsize=(8, 5))
plt.plot(list(avg_mious_per_n.keys()), list(avg_mious_per_n.values()), marker='o')
plt.title('Average mIoU vs Number of Masks (across dataset)')
plt.xlabel('Number of Masks')
plt.ylabel('Average mIoU')
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "average_mIoU_vs_n_masks.png", dpi=150)
plt.close()

# Plot histogram of number of objects
plt.figure(figsize=(8, 5))
plt.hist(num_objects_list, bins=20)
plt.title('Distribution of Number of Objects per Image')
plt.xlabel('Number of Objects')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "num_objects_distribution.png", dpi=150)
plt.close()

# Plot average correlations as a bar chart
plt.figure(figsize=(8, 5))
sorted_attrs = sorted(avg_correlations.keys(), key=lambda k: avg_correlations[k])
vals = [avg_correlations[a] for a in sorted_attrs]
plt.bar(sorted_attrs, vals)
plt.title('Average Correlations with Region Properties')
plt.xlabel('Property')
plt.ylabel('Average Spearman Correlation')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(output_dir / "average_correlations.png", dpi=150)
plt.close()

print(f"Plots saved to {output_dir}")
