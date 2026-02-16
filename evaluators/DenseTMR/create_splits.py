import json
import random

# Load the annotation file
with open('annotations.json', 'r') as f:
    data = json.load(f)

# Get all IDs and sort them
all_ids = sorted(data.keys())
print(f"Total samples: {len(all_ids)}")

# Shuffle for random split
random.seed(42)  # For reproducibility
shuffled_ids = all_ids.copy()
random.shuffle(shuffled_ids)

# Calculate split sizes
total = len(shuffled_ids)
train_size = int(total * 0.80)
test_size = int(total * 0.15)
val_size = total - train_size - test_size  # Remaining goes to val (~5%)

# Split the data
train_ids = sorted(shuffled_ids[:train_size])
test_ids = sorted(shuffled_ids[train_size:train_size + test_size])
val_ids = sorted(shuffled_ids[train_size + test_size:])

print(f"Train: {len(train_ids)} ({len(train_ids)/total*100:.1f}%)")
print(f"Test: {len(test_ids)} ({len(test_ids)/total*100:.1f}%)")
print(f"Val: {len(val_ids)} ({len(val_ids)/total*100:.1f}%)")

# Write all.txt
with open('splits/all.txt', 'w') as f:
    for id in sorted(all_ids):
        f.write(f"{id}\n")

# Write train.txt
with open('splits/train.txt', 'w') as f:
    for id in train_ids:
        f.write(f"{id}\n")

# Write test.txt
with open('splits/test.txt', 'w') as f:
    for id in test_ids:
        f.write(f"{id}\n")

# Write val.txt
with open('splits/val.txt', 'w') as f:
    for id in val_ids:
        f.write(f"{id}\n")

print("\nFiles created: all.txt, train.txt, test.txt, val.txt")