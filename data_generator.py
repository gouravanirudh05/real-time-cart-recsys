"""
Synthetic Food Delivery Dataset Generator for CSAO Rail Recommendation System
Generates realistic orders, users, restaurants, and items with natural combo patterns.

Outputs:
- users.csv
- restaurants.csv
- menu_items.csv
- orders.csv           (order-level metadata)
- order_items.csv      (items per order)
- user_item_interactions.csv  (for model training: positive & negative samples)
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import itertools
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Scale knobs ───────────────────────────────────────────────────────────────
N_USERS        = 2_000
N_RESTAURANTS  = 150
N_ORDERS       = 50_000
OUTPUT_DIR     = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Domain taxonomy ───────────────────────────────────────────────────────────
CUISINES = ["North Indian", "South Indian", "Chinese", "Italian", "Fast Food",
            "Biryani", "Desserts", "Beverages", "Healthy", "Street Food"]

DISH_FAMILIES = {
    "North Indian": ["Curry", "Dal", "Paneer Dish", "Bread", "Rice", "Raita", "Pickle"],
    "South Indian": ["Dosa", "Idli", "Sambar", "Chutney", "Rice", "Rasam", "Filter Coffee"],
    "Chinese":      ["Noodles", "Fried Rice", "Manchurian", "Spring Roll", "Soup", "Dimsum"],
    "Italian":      ["Pizza", "Pasta", "Garlic Bread", "Risotto", "Tiramisu"],
    "Fast Food":    ["Burger", "Fries", "Wrap", "Nuggets", "Milkshake", "Soft Drink"],
    "Biryani":      ["Biryani", "Raita", "Salan", "Shorba", "Kebab", "Gulab Jamun"],
    "Desserts":     ["Ice Cream", "Brownie", "Gulab Jamun", "Kheer", "Pastry", "Cake"],
    "Beverages":    ["Soft Drink", "Juice", "Lassi", "Chai", "Coffee", "Water"],
    "Healthy":      ["Salad", "Smoothie", "Wrap", "Oats", "Fruit Bowl", "Quinoa Bowl"],
    "Street Food":  ["Chaat", "Pav Bhaji", "Vada Pav", "Pani Puri", "Bhel Puri", "Samosa"],
}

# Natural combos: (dish_family_1, dish_family_2) pairs that make sense together
NATURAL_COMBOS = [
    ("Biryani", "Raita"), ("Biryani", "Salan"), ("Biryani", "Kebab"),
    ("Biryani", "Soft Drink"), ("Biryani", "Gulab Jamun"),
    ("Curry", "Bread"), ("Curry", "Rice"), ("Curry", "Dal"),
    ("Pizza", "Garlic Bread"), ("Pizza", "Soft Drink"), ("Pizza", "Pasta"),
    ("Burger", "Fries"), ("Burger", "Milkshake"), ("Burger", "Soft Drink"),
    ("Dosa", "Sambar"), ("Dosa", "Chutney"), ("Dosa", "Filter Coffee"),
    ("Noodles", "Manchurian"), ("Noodles", "Spring Roll"), ("Fried Rice", "Manchurian"),
    ("Salad", "Smoothie"), ("Salad", "Wrap"),
    ("Chaat", "Soft Drink"), ("Chaat", "Juice"),
    ("Paneer Dish", "Bread"), ("Paneer Dish", "Rice"), ("Paneer Dish", "Raita"),
    ("Ice Cream", "Brownie"), ("Pastry", "Coffee"), ("Cake", "Coffee"),
    ("Idli", "Sambar"), ("Idli", "Chutney"),
    ("Dal", "Rice"), ("Rasam", "Rice"),
]
NATURAL_COMBO_SET = set(NATURAL_COMBOS) | {(b, a) for a, b in NATURAL_COMBOS}

# Meal time windows
MEAL_TIMES = {
    "breakfast": (7, 10),
    "lunch":     (12, 15),
    "snack":     (16, 18),
    "dinner":    (19, 23),
    "late_night":(23, 2),
}

MEAL_TIME_CUISINE_WEIGHTS = {
    "breakfast":  {"South Indian": 0.35, "Healthy": 0.2, "Beverages": 0.2, "Street Food": 0.1, "Fast Food": 0.15},
    "lunch":      {"North Indian": 0.25, "Biryani": 0.25, "South Indian": 0.15, "Chinese": 0.15, "Fast Food": 0.1, "Italian": 0.1},
    "snack":      {"Street Food": 0.3, "Fast Food": 0.25, "Beverages": 0.2, "Desserts": 0.15, "Healthy": 0.1},
    "dinner":     {"North Indian": 0.25, "Biryani": 0.2, "Chinese": 0.2, "Italian": 0.15, "Fast Food": 0.1, "South Indian": 0.1},
    "late_night": {"Fast Food": 0.4, "Chinese": 0.25, "Biryani": 0.2, "Street Food": 0.15},
}

USER_SEGMENTS = ["budget", "regular", "premium", "occasional"]
CITIES = ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata"]
PRICE_RANGES = ["budget", "mid", "premium"]

# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATE USERS
# ─────────────────────────────────────────────────────────────────────────────
print("Generating users...")

def generate_users(n):
    segments = np.random.choice(USER_SEGMENTS, n, p=[0.3, 0.4, 0.2, 0.1])
    cities   = np.random.choice(CITIES, n)
    users = []
    for i in range(n):
        seg = segments[i]
        city = cities[i]
        # Avg order frequency per month depends on segment
        freq_map = {"budget": (4, 8), "regular": (8, 15), "premium": (15, 30), "occasional": (1, 4)}
        lo, hi = freq_map[seg]
        users.append({
            "user_id": f"U{i+1:05d}",
            "segment": seg,
            "city": city,
            "delivery_zone": f"{city}_Z{np.random.randint(1, 6)}",
            "preferred_cuisines": "|".join(np.random.choice(CUISINES, size=np.random.randint(2, 5), replace=False)),
            "is_veg": np.random.choice([True, False], p=[0.35, 0.65]),
            "avg_monthly_orders": np.random.randint(lo, hi),
            "price_sensitivity": np.random.choice(["low", "medium", "high"],
                                                   p=[0.3, 0.4, 0.3] if seg == "budget" else [0.5, 0.3, 0.2] if seg == "premium" else [0.4, 0.4, 0.2]),
            "account_age_days": np.random.randint(30, 1000),
        })
    return pd.DataFrame(users)

users_df = generate_users(N_USERS)
users_df.to_csv(f"{OUTPUT_DIR}/users.csv", index=False)
print(f"  → {len(users_df)} users")

# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERATE RESTAURANTS
# ─────────────────────────────────────────────────────────────────────────────
print("Generating restaurants...")

def generate_restaurants(n):
    rests = []
    for i in range(n):
        cuisine = np.random.choice(CUISINES)
        price   = np.random.choice(PRICE_RANGES, p=[0.4, 0.4, 0.2])
        city    = np.random.choice(CITIES)
        top_families = random.sample(DISH_FAMILIES[cuisine], min(5, len(DISH_FAMILIES[cuisine])))
        rests.append({
            "restaurant_id": f"R{i+1:04d}",
            "name": f"{cuisine} Place {i+1}",
            "cuisine": cuisine,
            "price_range": price,
            "city": city,
            "zone": f"{city}_Z{np.random.randint(1, 6)}",
            "rating": round(np.random.uniform(3.5, 5.0), 1),
            "is_chain": np.random.choice([True, False], p=[0.3, 0.7]),
            "top_dish_families": "|".join(top_families),
            "avg_delivery_time_min": np.random.randint(20, 60),
            "total_orders_last_30d": np.random.randint(50, 3000),
        })
    return pd.DataFrame(rests)

restaurants_df = generate_restaurants(N_RESTAURANTS)
restaurants_df.to_csv(f"{OUTPUT_DIR}/restaurants.csv", index=False)
print(f"  → {len(restaurants_df)} restaurants")

# ─────────────────────────────────────────────────────────────────────────────
# 3. GENERATE MENU ITEMS
# ─────────────────────────────────────────────────────────────────────────────
print("Generating menu items...")

DISH_TEMPLATES = {
    "Curry":        ["Butter Chicken", "Palak Paneer", "Chicken Masala", "Lamb Curry", "Mixed Veg Curry"],
    "Dal":          ["Dal Makhani", "Dal Tadka", "Dal Fry", "Chana Masala"],
    "Paneer Dish":  ["Paneer Tikka Masala", "Shahi Paneer", "Kadai Paneer", "Matar Paneer"],
    "Bread":        ["Butter Naan", "Roti", "Paratha", "Garlic Naan", "Puri"],
    "Rice":         ["Steamed Rice", "Jeera Rice", "Veg Pulao", "Egg Fried Rice"],
    "Raita":        ["Boondi Raita", "Cucumber Raita", "Mixed Raita"],
    "Pickle":       ["Mango Pickle", "Mixed Pickle", "Lime Pickle"],
    "Biryani":      ["Chicken Biryani", "Mutton Biryani", "Veg Biryani", "Egg Biryani", "Prawn Biryani"],
    "Salan":        ["Mirchi Ka Salan", "Shorba", "Biryani Gravy"],
    "Kebab":        ["Seekh Kebab", "Chicken 65", "Reshmi Kebab", "Boti Kebab"],
    "Gulab Jamun":  ["Gulab Jamun (2 pcs)", "Gulab Jamun (4 pcs)"],
    "Dosa":         ["Masala Dosa", "Plain Dosa", "Onion Dosa", "Rava Dosa", "Ghee Roast Dosa"],
    "Idli":         ["Idli (2 pcs)", "Idli (4 pcs)", "Mini Idli"],
    "Sambar":       ["Sambar Bowl", "Drumstick Sambar"],
    "Chutney":      ["Coconut Chutney", "Tomato Chutney", "Mint Chutney"],
    "Filter Coffee":["Filter Coffee", "South Indian Coffee"],
    "Rasam":        ["Tomato Rasam", "Pepper Rasam"],
    "Noodles":      ["Veg Hakka Noodles", "Chicken Noodles", "Schezwan Noodles"],
    "Fried Rice":   ["Veg Fried Rice", "Egg Fried Rice", "Chicken Fried Rice"],
    "Manchurian":   ["Gobi Manchurian", "Chicken Manchurian", "Veg Manchurian"],
    "Spring Roll":  ["Veg Spring Roll", "Chicken Spring Roll"],
    "Soup":         ["Sweet Corn Soup", "Hot and Sour Soup", "Manchow Soup"],
    "Dimsum":       ["Veg Dimsum (6 pcs)", "Chicken Dimsum (6 pcs)"],
    "Pizza":        ["Margherita Pizza", "Pepperoni Pizza", "BBQ Chicken Pizza", "Veggie Supreme"],
    "Pasta":        ["Arrabiata Pasta", "Alfredo Pasta", "Pesto Pasta", "Mac and Cheese"],
    "Garlic Bread": ["Garlic Bread", "Cheesy Garlic Bread"],
    "Tiramisu":     ["Tiramisu"],
    "Risotto":      ["Mushroom Risotto", "Chicken Risotto"],
    "Burger":       ["Veg Burger", "Chicken Burger", "Zinger Burger", "Double Patty Burger"],
    "Fries":        ["Regular Fries", "Cheese Fries", "Peri Peri Fries"],
    "Wrap":         ["Chicken Wrap", "Paneer Wrap", "Veg Wrap"],
    "Nuggets":      ["Chicken Nuggets (6 pcs)", "Chicken Nuggets (12 pcs)"],
    "Milkshake":    ["Chocolate Milkshake", "Strawberry Milkshake", "Vanilla Milkshake"],
    "Soft Drink":   ["Coke (300ml)", "Pepsi (300ml)", "Sprite (300ml)", "Limca"],
    "Ice Cream":    ["Vanilla Scoop", "Chocolate Scoop", "Butterscotch Scoop", "Sundae"],
    "Brownie":      ["Chocolate Brownie", "Walnut Brownie"],
    "Kheer":        ["Rice Kheer", "Vermicelli Kheer"],
    "Pastry":       ["Chocolate Pastry", "Black Forest Pastry"],
    "Cake":         ["Chocolate Truffle Slice", "Red Velvet Slice"],
    "Juice":        ["Fresh Orange Juice", "Watermelon Juice", "Mixed Fruit Juice"],
    "Lassi":        ["Sweet Lassi", "Salted Lassi", "Mango Lassi"],
    "Chai":         ["Masala Chai", "Ginger Chai"],
    "Coffee":       ["Cappuccino", "Latte", "Espresso"],
    "Water":        ["Mineral Water (500ml)"],
    "Salad":        ["Caesar Salad", "Greek Salad", "Garden Salad"],
    "Smoothie":     ["Banana Smoothie", "Mixed Berry Smoothie", "Green Detox Smoothie"],
    "Oats":         ["Masala Oats", "Fruit Oats"],
    "Fruit Bowl":   ["Seasonal Fruit Bowl"],
    "Quinoa Bowl":  ["Protein Quinoa Bowl", "Veggie Quinoa Bowl"],
    "Chaat":        ["Papdi Chaat", "Aloo Chaat", "Dahi Puri"],
    "Pav Bhaji":    ["Pav Bhaji (2 pav)", "Pav Bhaji (4 pav)"],
    "Vada Pav":     ["Vada Pav", "Cheese Vada Pav"],
    "Pani Puri":    ["Pani Puri (6 pcs)"],
    "Bhel Puri":    ["Bhel Puri"],
    "Samosa":       ["Samosa (2 pcs)", "Samosa (4 pcs)"],
    "Shorba":       ["Chicken Shorba", "Veg Shorba"],
}

PRICE_RANGES_MAP = {
    "budget":  (50, 150),
    "mid":     (100, 350),
    "premium": (300, 800),
}

items = []
item_id = 1
for _, rest in restaurants_df.iterrows():
    cuisine = rest["cuisine"]
    families = DISH_FAMILIES[cuisine]
    price_lo, price_hi = PRICE_RANGES_MAP[rest["price_range"]]
    for fam in families:
        templates = DISH_TEMPLATES.get(fam, [f"{fam} Special"])
        for dish_name in templates:
            is_veg = any(k in dish_name.lower() for k in ["veg", "paneer", "aloo", "corn", "mushroom",
                                                            "fruit", "oats", "salad", "water", "juice",
                                                            "chai", "coffee", "lassi", "smoothie", "gulab"])
            items.append({
                "item_id": f"I{item_id:06d}",
                "restaurant_id": rest["restaurant_id"],
                "name": dish_name,
                "dish_family": fam,
                "cuisine": cuisine,
                "price": round(np.random.uniform(price_lo, price_hi) / 5) * 5,
                "is_veg": is_veg,
                "is_addon_eligible": True,
                "popularity_score": round(np.random.uniform(0.3, 1.0), 2),
                "avg_rating": round(np.random.uniform(3.5, 5.0), 1),
            })
            item_id += 1

items_df = pd.DataFrame(items)
items_df.to_csv(f"{OUTPUT_DIR}/menu_items.csv", index=False)
print(f"  → {len(items_df)} menu items")

# Build a fast lookup: restaurant_id → list of items
rest_items = items_df.groupby("restaurant_id").apply(lambda x: x.to_dict("records")).to_dict()

# ─────────────────────────────────────────────────────────────────────────────
# 4. GENERATE ORDERS + ORDER ITEMS
# ─────────────────────────────────────────────────────────────────────────────
print("Generating orders and order items...")

def get_meal_time(hour):
    if 7 <= hour < 11:   return "breakfast"
    if 11 <= hour < 15:  return "lunch"
    if 15 <= hour < 18:  return "snack"
    if 18 <= hour < 23:  return "dinner"
    return "late_night"

def pick_cuisine_for_meal(meal_time, user_prefs):
    weights_map = MEAL_TIME_CUISINE_WEIGHTS.get(meal_time, {c: 1/len(CUISINES) for c in CUISINES})
    available = [c for c in user_prefs if c in weights_map]
    if not available:
        available = list(weights_map.keys())
    w = np.array([weights_map.get(c, 0.05) for c in available], dtype=float)
    w /= w.sum()
    return np.random.choice(available, p=w)

def build_cart(user, rest_item_list, meal_time, n_items_range=(1, 4)):
    """
    Build a cart that naturally follows food combo patterns.
    With 70% probability, uses natural combos as seeds.
    """
    by_family = {}
    for it in rest_item_list:
        by_family.setdefault(it["dish_family"], []).append(it)

    n_items = np.random.randint(*n_items_range)
    chosen = []

    if len(by_family) < 2:
        # fallback: just pick random
        pool = rest_item_list.copy()
        random.shuffle(pool)
        return pool[:n_items]

    # Try to seed with a natural combo
    if np.random.rand() < 0.70:
        available_families = list(by_family.keys())
        # Find valid natural combos possible in this restaurant
        valid_combos = [(f1, f2) for f1, f2 in NATURAL_COMBOS
                        if f1 in by_family and f2 in by_family and f1 != f2]
        if valid_combos:
            f1, f2 = random.choice(valid_combos)
            chosen.append(random.choice(by_family[f1]))
            chosen.append(random.choice(by_family[f2]))

    # Fill remaining slots
    added_families = {it["dish_family"] for it in chosen}
    remaining_families = [f for f in by_family if f not in added_families]
    random.shuffle(remaining_families)
    for fam in remaining_families:
        if len(chosen) >= n_items:
            break
        if np.random.rand() < 0.5:  # don't always add more
            chosen.append(random.choice(by_family[fam]))

    if not chosen:
        chosen = [random.choice(rest_item_list)]

    return chosen[:n_items]


start_date = datetime.now() - timedelta(days=180)

orders = []
order_items_rows = []
order_id = 1

# Give each user a random restaurant affinity
user_rest_affinity = {}
for _, u in users_df.iterrows():
    prefs = set(u["preferred_cuisines"].split("|"))
    matching_rests = restaurants_df[restaurants_df["cuisine"].isin(prefs)]["restaurant_id"].tolist()
    if not matching_rests:
        matching_rests = restaurants_df["restaurant_id"].tolist()
    user_rest_affinity[u["user_id"]] = matching_rests

# Map user_id to segment
user_segment_map = users_df.set_index("user_id")["segment"].to_dict()
user_prefs_map    = users_df.set_index("user_id")["preferred_cuisines"].apply(lambda x: x.split("|")).to_dict()
user_city_map     = users_df.set_index("user_id")["city"].to_dict()
user_veg_map      = users_df.set_index("user_id")["is_veg"].to_dict()

user_ids = users_df["user_id"].tolist()

for _ in range(N_ORDERS):
    uid = random.choice(user_ids)
    seg = user_segment_map[uid]
    city = user_city_map[uid]
    is_veg = user_veg_map[uid]

    # Pick time
    order_dt = start_date + timedelta(
        days=np.random.randint(0, 180),
        hours=np.random.randint(0, 24),
        minutes=np.random.randint(0, 60)
    )
    hour = order_dt.hour
    meal_time = get_meal_time(hour)

    # Pick restaurant in same city preferably
    city_rests = restaurants_df[restaurants_df["city"] == city]["restaurant_id"].tolist()
    affinity_rests = user_rest_affinity[uid]
    combined = list(set(city_rests) & set(affinity_rests))
    if not combined:
        combined = affinity_rests or restaurants_df["restaurant_id"].tolist()
    rest_id = random.choice(combined)

    if rest_id not in rest_items or not rest_items[rest_id]:
        continue

    item_pool = rest_items[rest_id]

    # Veg users only order veg items
    if is_veg:
        item_pool = [it for it in item_pool if it["is_veg"]] or item_pool

    if not item_pool:
        continue

    n_range = (1, 3) if seg == "budget" else (2, 5) if seg in ["regular", "occasional"] else (3, 7)
    cart_items = build_cart({"is_veg": is_veg}, item_pool, meal_time, n_items_range=n_range)

    total_value = sum(it["price"] for it in cart_items)
    rest_info = restaurants_df[restaurants_df["restaurant_id"] == rest_id].iloc[0]

    orders.append({
        "order_id": f"O{order_id:07d}",
        "user_id": uid,
        "restaurant_id": rest_id,
        "order_datetime": order_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "day_of_week": order_dt.strftime("%A"),
        "hour": hour,
        "meal_time": meal_time,
        "total_value": total_value,
        "num_items": len(cart_items),
        "city": city,
        "delivery_zone": f"{city}_Z{np.random.randint(1, 6)}",
        "user_segment": seg,
        "cuisine": rest_info["cuisine"],
        "restaurant_price_range": rest_info["price_range"],
        "order_status": "completed",
    })

    for it in cart_items:
        order_items_rows.append({
            "order_id": f"O{order_id:07d}",
            "user_id": uid,
            "restaurant_id": rest_id,
            "item_id": it["item_id"],
            "item_name": it["name"],
            "dish_family": it["dish_family"],
            "cuisine": it["cuisine"],
            "price": it["price"],
            "is_veg": it["is_veg"],
            "quantity": 1,
        })

    order_id += 1

orders_df      = pd.DataFrame(orders)
order_items_df = pd.DataFrame(order_items_rows)

orders_df.to_csv(f"{OUTPUT_DIR}/orders.csv", index=False)
order_items_df.to_csv(f"{OUTPUT_DIR}/order_items.csv", index=False)
print(f"  → {len(orders_df)} orders, {len(order_items_df)} order-item rows")

# ─────────────────────────────────────────────────────────────────────────────
# 5. BUILD USER-ITEM INTERACTION TABLE (Training data for recommendation model)
# ─────────────────────────────────────────────────────────────────────────────
print("Building user-item interactions (positive + negative samples)...")

# For each order, create pairwise positive samples (items that co-occurred)
# and negative samples (items from same restaurant that were NOT ordered together)

interactions = []

for oid, grp in order_items_df.groupby("order_id"):
    order_row = orders_df[orders_df["order_id"] == oid].iloc[0]
    uid  = order_row["user_id"]
    rid  = order_row["restaurant_id"]
    mt   = order_row["meal_time"]
    dow  = order_row["day_of_week"]
    hour = order_row["hour"]
    seg  = order_row["user_segment"]
    city = order_row["city"]
    tv   = order_row["total_value"]

    items_in_order = grp.to_dict("records")

    if len(items_in_order) < 2:
        # Single-item order: treat as anchor, no pair
        continue

    # Positive samples: all pairs in cart
    for it1, it2 in itertools.combinations(items_in_order, 2):
        if it1["dish_family"] == it2["dish_family"]:
            continue
        interactions.append({
            "user_id": uid,
            "order_id": oid,
            "restaurant_id": rid,
            "anchor_item_id": it1["item_id"],
            "anchor_dish_family": it1["dish_family"],
            "anchor_price": it1["price"],
            "candidate_item_id": it2["item_id"],
            "candidate_dish_family": it2["dish_family"],
            "candidate_price": it2["price"],
            "label": 1,
            "is_natural_combo": int((it1["dish_family"], it2["dish_family"]) in NATURAL_COMBO_SET),
            "meal_time": mt,
            "day_of_week": dow,
            "hour": hour,
            "user_segment": seg,
            "city": city,
            "cart_total_value": tv,
        })

    # Negative samples: replace one item with a random restaurant item not in cart
    cart_ids = {it["item_id"] for it in items_in_order}
    rest_pool = [it for it in rest_items.get(rid, []) if it["item_id"] not in cart_ids]
    if rest_pool and items_in_order:
        anchor = random.choice(items_in_order)
        neg    = random.choice(rest_pool)
        if anchor["dish_family"] != neg["dish_family"]:
            interactions.append({
                "user_id": uid,
                "order_id": oid,
                "restaurant_id": rid,
                "anchor_item_id": anchor["item_id"],
                "anchor_dish_family": anchor["dish_family"],
                "anchor_price": anchor["price"],
                "candidate_item_id": neg["item_id"],
                "candidate_dish_family": neg["dish_family"],
                "candidate_price": neg["price"],
                "label": 0,
                "is_natural_combo": int((anchor["dish_family"], neg["dish_family"]) in NATURAL_COMBO_SET),
                "meal_time": mt,
                "day_of_week": dow,
                "hour": hour,
                "user_segment": seg,
                "city": city,
                "cart_total_value": tv,
            })

interactions_df = pd.DataFrame(interactions)
interactions_df.to_csv(f"{OUTPUT_DIR}/user_item_interactions.csv", index=False)
print(f"  → {len(interactions_df)} interaction rows ({interactions_df['label'].sum()} positive, {(interactions_df['label']==0).sum()} negative)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. USER HISTORY FEATURES (aggregated per user for model features)
# ─────────────────────────────────────────────────────────────────────────────
print("Building user history feature table...")

user_history = order_items_df.groupby("user_id").agg(
    total_orders    = ("order_id", "nunique"),
    unique_families = ("dish_family", "nunique"),
    unique_cuisines = ("cuisine", "nunique"),
    total_spent     = ("price", "sum"),
    avg_item_price  = ("price", "mean"),
    pct_veg         = ("is_veg", "mean"),
).reset_index()

user_history["diversity_factor"] = user_history["unique_families"] / user_history["total_orders"].clip(lower=1)
user_history["avg_spent_per_order"] = user_history["total_spent"] / user_history["total_orders"].clip(lower=1)

# Top dish family per user
top_family = (order_items_df.groupby(["user_id", "dish_family"])
               .size()
               .reset_index(name="count")
               .sort_values("count", ascending=False)
               .drop_duplicates("user_id")[["user_id", "dish_family"]]
               .rename(columns={"dish_family": "top_dish_family"}))

user_history = user_history.merge(top_family, on="user_id", how="left")
user_history = user_history.merge(users_df[["user_id", "segment", "city", "is_veg", "price_sensitivity"]], on="user_id")
user_history.to_csv(f"{OUTPUT_DIR}/user_history_features.csv", index=False)
print(f"  → {len(user_history)} user history rows")

# ─────────────────────────────────────────────────────────────────────────────
# 7. RESTAURANT PERFORMANCE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("Building restaurant performance features...")

rest_perf = order_items_df.groupby("restaurant_id").agg(
    total_orders     = ("order_id", "nunique"),
    total_items_sold = ("item_id", "count"),
    unique_items     = ("item_id", "nunique"),
    avg_item_price   = ("price", "mean"),
).reset_index()

top_rest_family = (order_items_df.groupby(["restaurant_id", "dish_family"])
                   .size()
                   .reset_index(name="cnt")
                   .sort_values("cnt", ascending=False)
                   .groupby("restaurant_id")
                   .head(5)
                   .groupby("restaurant_id")["dish_family"]
                   .apply(lambda x: "|".join(x))
                   .reset_index()
                   .rename(columns={"dish_family": "top_5_dish_families"}))

rest_perf = rest_perf.merge(top_rest_family, on="restaurant_id", how="left")
rest_perf = rest_perf.merge(restaurants_df[["restaurant_id", "cuisine", "price_range", "rating", "city"]], on="restaurant_id")
rest_perf.to_csv(f"{OUTPUT_DIR}/restaurant_performance_features.csv", index=False)
print(f"  → {len(rest_perf)} restaurant rows")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n✅ Dataset generation complete!")
print(f"   Output directory: {OUTPUT_DIR}/")
print(f"""
Files created:
  users.csv                          - {len(users_df)} users with segment, preferences, city
  restaurants.csv                    - {len(restaurants_df)} restaurants with cuisine, price range
  menu_items.csv                     - {len(items_df)} menu items with dish family, price, veg flag
  orders.csv                         - {len(orders_df)} orders with meal time, cart value
  order_items.csv                    - {len(order_items_df)} order-item rows
  user_item_interactions.csv         - {len(interactions_df)} training samples (label=1/0)
  user_history_features.csv          - Aggregated user behavior features
  restaurant_performance_features.csv- Aggregated restaurant performance features

How to use for the recommendation model:
  • user_item_interactions.csv is your core training data (binary classification)
    - anchor_item_id  = item already in cart
    - candidate_item_id = potential add-on to recommend
    - label = 1 (user added it) / 0 (negative sample)
  • Join with user_history_features, restaurant_performance_features, menu_items
    for the full feature set.
  • Use temporal split on order_datetime for train/val/test.
  • Diversity factor is pre-computed for FoodNet-style monotonic constraint.
""")