"""Quick check of data to understand the indexing issue"""
import numpy as np

dir_str = 'Movielens'
train_edge = np.load(dir_str+'/train_sample.npy', allow_pickle=True)
v_feat = np.load(dir_str+'/v_feat_sample.npy', allow_pickle=True)

print(f"Train edge shape: {train_edge.shape}")
print(f"V feat shape: {v_feat.shape}")
print(f"Train edge sample (first 10 rows):\n{train_edge[:10]}")
print(f"\nUser range: [{train_edge[:, 0].min()}, {train_edge[:, 0].max()}]")
print(f"Item range: [{train_edge[:, 1].min()}, {train_edge[:, 1].max()}]")
print(f"\nExpected item range based on features: [?, {v_feat.shape[0] - 1}]")
print(f"\nIf items are offset by num_user, expected range: [num_user, num_user + {v_feat.shape[0] - 1}]")
