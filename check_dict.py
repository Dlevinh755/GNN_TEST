"""Check user_item_dict format"""
import numpy as np

dir_str = 'Movielens'
user_item_dict = np.load(dir_str+'/user_item_dict_sample.npy', allow_pickle=True).item()

# Check a few entries
print("Sample entries from user_item_dict:")
for i, (user, items) in enumerate(list(user_item_dict.items())[:5]):
    print(f"User {user}: items = {list(items)[:10]}...")  # Show first 10 items
    
# Check the range
all_items = set()
for items in user_item_dict.values():
    all_items.update(items)

print(f"\nTotal unique items in dict: {len(all_items)}")
print(f"Item range: [{min(all_items)}, {max(all_items)}]")
