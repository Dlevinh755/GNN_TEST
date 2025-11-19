"""
Debug script to verify data dimensions and index ranges
"""
import numpy as np
import torch

# Load the data
dir_str = 'Movielens'

# Load features
v_feat = np.load(dir_str+'/v_feat_sample.npy', allow_pickle=True)
a_feat = np.load(dir_str+'/a_feat_sample.npy', allow_pickle=True)
t_feat = np.load(dir_str+'/t_feat_sample.npy', allow_pickle=True)

# Load edge data
train_edge = np.load(dir_str+'/train_sample.npy', allow_pickle=True)
user_item_dict = np.load(dir_str+'/user_item_dict_sample.npy', allow_pickle=True).item()

print("=" * 60)
print("DATA DIMENSIONS")
print("=" * 60)
print(f"v_feat shape: {v_feat.shape}")
print(f"a_feat shape: {a_feat.shape}")
print(f"t_feat shape: {t_feat.shape}")
print(f"train_edge shape: {train_edge.shape}")
print(f"Number of users in dict: {len(user_item_dict)}")

print("\n" + "=" * 60)
print("INDEX RANGES IN TRAINING DATA")
print("=" * 60)
print(f"User indices - min: {train_edge[:, 0].min()}, max: {train_edge[:, 0].max()}")
print(f"Item indices - min: {train_edge[:, 1].min()}, max: {train_edge[:, 1].max()}")

print("\n" + "=" * 60)
print("EXPECTED RANGES")
print("=" * 60)
num_users = train_edge[:, 0].max() + 1
num_items_from_features = v_feat.shape[0]
print(f"Number of unique users: {num_users}")
print(f"Number of items (from features): {num_items_from_features}")
print(f"User indices should be in range: [0, {num_users - 1}]")
print(f"Item indices should be in range: [{num_users}, {num_users + num_items_from_features - 1}]")

print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)
max_item_in_data = train_edge[:, 1].max()
expected_max_item = num_users + num_items_from_features - 1
if max_item_in_data > expected_max_item:
    print(f"❌ ERROR: Max item index {max_item_in_data} exceeds expected max {expected_max_item}")
    print(f"   Difference: {max_item_in_data - expected_max_item}")
else:
    print(f"✓ OK: All item indices are within valid range")

# Check user_item_dict
all_items_in_dict = set()
for items in user_item_dict.values():
    all_items_in_dict.update(items)

min_item_dict = min(all_items_in_dict)
max_item_dict = max(all_items_in_dict)
print(f"\nItems in user_item_dict - min: {min_item_dict}, max: {max_item_dict}")
if max_item_dict > expected_max_item:
    print(f"❌ ERROR: Max item in dict {max_item_dict} exceeds expected max {expected_max_item}")
else:
    print(f"✓ OK: All items in dict are within valid range")
