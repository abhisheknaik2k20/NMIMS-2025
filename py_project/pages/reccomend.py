import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛒",
    layout="wide"
)

# Function to generate synthetic user data
def generate_users(num_users=100):
    users = []
    
    # Demographics
    age_ranges = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    genders = ["Male", "Female", "Other"]
    locations = ["North", "South", "East", "West", "Central"]
    
    # Generate user profiles
    for user_id in range(1, num_users + 1):
        # Basic demographics
        age = random.choice(age_ranges)
        gender = random.choice(genders)
        location = random.choice(locations)
        
        # Activity metrics
        signup_date = datetime.now() - timedelta(days=random.randint(1, 365))
        last_active = signup_date + timedelta(days=random.randint(0, (datetime.now() - signup_date).days))
        days_active = max(1, (last_active - signup_date).days)
        
        # Engagement metrics (scaled relative to days_active)
        visit_frequency = round(random.uniform(0.1, 1.0), 2)  # Visits per day
        avg_session_duration = round(random.uniform(1, 30), 1)  # Minutes
        
        # Purchase behavior
        purchase_frequency = round(random.uniform(0, 0.3), 2)  # Purchases per day
        avg_order_value = round(random.uniform(10, 200), 2)  # Currency
        
        # Preferences - scale from 1-10
        price_sensitivity = random.randint(1, 10)
        brand_loyalty = random.randint(1, 10)
        trend_follower = random.randint(1, 10)
        
        # Category preferences (random weighting)
        categories = ["Electronics", "Clothing", "Home", "Beauty", "Sports", "Books", "Food"]
        category_prefs = {}
        for category in categories:
            category_prefs[f"{category}_pref"] = random.randint(1, 10)
        
        # User profile
        user = {
            "user_id": user_id,
            "age_group": age,
            "gender": gender,
            "location": location,
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "last_active": last_active.strftime("%Y-%m-%d"),
            "days_active": days_active,
            "visit_frequency": visit_frequency,
            "avg_session_duration": avg_session_duration,
            "purchase_frequency": purchase_frequency,
            "avg_order_value": avg_order_value,
            "price_sensitivity": price_sensitivity,
            "brand_loyalty": brand_loyalty,
            "trend_follower": trend_follower,
            **category_prefs
        }
        
        users.append(user)
    
    return pd.DataFrame(users)

# Function to generate synthetic product data
def generate_products(num_products=200):
    products = []
    
    # Product attributes
    categories = ["Electronics", "Clothing", "Home", "Beauty", "Sports", "Books", "Food"]
    
    # List of product names by category
    product_names = {
        "Electronics": ["Smartphone", "Laptop", "Headphones", "Tablet", "Smart Watch", "Camera", "Speaker", "TV", "Gaming Console", "Drone"],
        "Clothing": ["T-Shirt", "Jeans", "Dress", "Jacket", "Sweater", "Hoodie", "Sneakers", "Boots", "Hat", "Socks"],
        "Home": ["Sofa", "Bed", "Table", "Chair", "Lamp", "Rug", "Curtains", "Pillow", "Blender", "Toaster"],
        "Beauty": ["Moisturizer", "Sunscreen", "Cleanser", "Shampoo", "Conditioner", "Lipstick", "Foundation", "Mascara", "Perfume", "Body Lotion"],
        "Sports": ["Running Shoes", "Yoga Mat", "Dumbbells", "Tennis Racket", "Basketball", "Bike", "Helmet", "Treadmill", "Swimming Goggles", "Golf Clubs"],
        "Books": ["Novel", "Cookbook", "Self-Help", "Biography", "Science", "History", "Fantasy", "Mystery", "Children's Book", "Business Book"],
        "Food": ["Coffee", "Tea", "Chocolate", "Pasta", "Rice", "Cereal", "Snacks", "Cookies", "Bread", "Cheese"]
    }
    
    brands = {
        "Electronics": ["Apple", "Samsung", "Sony", "LG", "Bose", "Dell", "HP", "Lenovo", "Microsoft", "Canon"],
        "Clothing": ["Nike", "Adidas", "Zara", "H&M", "Levi's", "Gap", "Under Armour", "Ralph Lauren", "Gucci", "Puma"],
        "Home": ["IKEA", "Ashley", "Wayfair", "Crate & Barrel", "West Elm", "Pottery Barn", "HomeGoods", "Target", "Walmart", "Bed Bath & Beyond"],
        "Beauty": ["L'Oreal", "Neutrogena", "Dove", "Olay", "Maybelline", "MAC", "Estee Lauder", "Clinique", "Nivea", "CeraVe"],
        "Sports": ["Nike", "Adidas", "Under Armour", "The North Face", "Columbia", "Patagonia", "Wilson", "Callaway", "Yeti", "Reebok"],
        "Books": ["Penguin", "HarperCollins", "Simon & Schuster", "Random House", "Macmillan", "Hachette", "Scholastic", "Wiley", "Oxford", "Dover"],
        "Food": ["Nestle", "Kraft", "Kellogg's", "General Mills", "Pepsi", "Coca-Cola", "Hershey's", "Mars", "Danone", "Mondelez"]
    }
    
    # Create products
    for product_id in range(1, num_products + 1):
        category = random.choice(categories)
        name = random.choice(product_names[category])
        brand = random.choice(brands[category])
        
        # Product details
        color_options = ["Red", "Blue", "Black", "White", "Green", "Yellow", "Purple", "Brown", "Gray", "Orange", "Pink", "Silver", "Gold"]
        
        # Release date (within last 2 years)
        release_date = datetime.now() - timedelta(days=random.randint(1, 730))
        
        # Pricing
        base_price = 0
        if category == "Electronics":
            base_price = random.uniform(50, 2000)
        elif category == "Clothing":
            base_price = random.uniform(10, 200)
        elif category == "Home":
            base_price = random.uniform(20, 1000)
        elif category == "Beauty":
            base_price = random.uniform(5, 100)
        elif category == "Sports":
            base_price = random.uniform(10, 500)
        elif category == "Books":
            base_price = random.uniform(5, 50)
        elif category == "Food":
            base_price = random.uniform(2, 30)
        
        # Round price to 2 decimals
        price = round(base_price, 2)
        
        # Generate a sale price for some products
        on_sale = random.random() < 0.3  # 30% of products are on sale
        sale_price = round(price * random.uniform(0.5, 0.9), 2) if on_sale else None
        
        # Inventory status
        inventory_status = random.choice(["In Stock", "Low Stock", "Out of Stock"])
        inventory_count = 0
        if inventory_status == "In Stock":
            inventory_count = random.randint(10, 100)
        elif inventory_status == "Low Stock":
            inventory_count = random.randint(1, 9)
        else:
            inventory_count = 0
        
        # Attributes
        attributes = {}
        attributes["color"] = random.sample(color_options, random.randint(1, 5))
        
        # Ratings & Reviews
        avg_rating = round(random.uniform(1, 5), 1)
        num_reviews = random.randint(0, 500)
        
        # Tagging - on a scale of 1-10
        product_tags = {
            "premium": random.randint(1, 10),
            "budget": random.randint(1, 10),
            "trending": random.randint(1, 10),
            "eco_friendly": random.randint(1, 10),
            "limited_edition": random.randint(1, 10),
            "popular": random.randint(1, 10),
        }
        
        # Create product entry
        product = {
            "product_id": product_id,
            "name": f"{brand} {name}",
            "category": category,
            "brand": brand,
            "price": price,
            "sale_price": sale_price,
            "on_sale": on_sale,
            "release_date": release_date.strftime("%Y-%m-%d"),
            "days_since_release": (datetime.now() - release_date).days,
            "avg_rating": avg_rating,
            "num_reviews": num_reviews,
            "inventory_status": inventory_status,
            "inventory_count": inventory_count,
            **product_tags
        }
        
        products.append(product)
    
    return pd.DataFrame(products)

# Function to generate synthetic interaction data
def generate_interactions(users_df, products_df, min_per_user=3, max_per_user=20):
    interactions = []
    
    # Interaction types with weights
    interaction_types = ["view", "cart", "purchase", "save", "review"]
    interaction_weights = [0.6, 0.2, 0.1, 0.05, 0.05]  # probability distribution
    
    # For each user, generate some interactions
    for _, user in users_df.iterrows():
        user_id = user["user_id"]
        days_active = user["days_active"]
        
        # Determine number of interactions for this user
        # Ensure min_per_user is not greater than max_per_user
        safe_min = min(min_per_user, max_per_user)
        safe_max = max(min_per_user, max_per_user)
        
        # Scale max interactions based on days active, but ensure it's at least safe_min
        adjusted_max = min(safe_max, max(safe_min, days_active))
        num_interactions = random.randint(safe_min, adjusted_max)
        
        # Select products for this user
        # Weight products by how well they match user preferences
        product_weights = []
        
        for _, product in products_df.iterrows():
            # Base weight
            weight = 1.0
            
            # Adjust weight based on category preference
            category_pref = user.get(f"{product['category']}_pref", 5)  # default to middle if not found
            weight *= (category_pref / 5)  # scale by category preference
            
            # Adjust for price sensitivity
            if user["price_sensitivity"] > 5:  # price sensitive user
                if product["price"] > 100:
                    weight *= 0.5  # less likely to interact with expensive products
                else:
                    weight *= 1.5  # more likely to interact with cheaper products
            
            # Adjust for brand loyalty
            if user["brand_loyalty"] > 7:  # brand loyal user
                # Simulate having a "preferred brand"
                preferred_brand = product["brand"][:1]  # use first letter as a simple way to create preferred brands
                if product["brand"].startswith(preferred_brand):
                    weight *= 2.0  # much more likely to interact with preferred brand
            
            # Adjust for trend following
            if user["trend_follower"] > 7:  # trend follower
                if product["trending"] > 7:
                    weight *= 2.0  # more likely to interact with trending products
            
            product_weights.append(max(0.1, weight))  # ensure minimum weight
        
        # Normalize weights
        total_weight = sum(product_weights)
        product_weights = [w/total_weight for w in product_weights]
        
        # Sample products based on weights
        product_indices = np.random.choice(
            range(len(products_df)), 
            size=min(num_interactions, len(products_df)), 
            replace=False,  # no duplicate products
            p=product_weights
        )
        
        # Create interactions for the sampled products
        for idx in product_indices:
            product = products_df.iloc[idx]
            
            # Determine interaction type based on weights
            interaction_type = np.random.choice(interaction_types, p=interaction_weights)
            
            # Timestamp within user's active period
            signup_date = datetime.strptime(user["signup_date"], "%Y-%m-%d")
            last_active = datetime.strptime(user["last_active"], "%Y-%m-%d")
            
            # Ensure there's at least a 1-day window
            if (last_active - signup_date).days < 1:
                last_active = signup_date + timedelta(days=1)
                
            interaction_date = signup_date + timedelta(
                days=random.randint(0, (last_active - signup_date).days)
            )
            
            # Create interaction
            interaction = {
                "user_id": user_id,
                "product_id": product["product_id"],
                "interaction_type": interaction_type,
                "timestamp": interaction_date.strftime("%Y-%m-%d %H:%M:%S"),
                "interaction_value": 1  # base value
            }
            
            # Add interaction-specific attributes
            if interaction_type == "purchase":
                # Apply sale price if available
                if product["on_sale"] and product["sale_price"] is not None:
                    interaction["price_paid"] = product["sale_price"]
                else:
                    interaction["price_paid"] = product["price"]
                
                # Quantity purchased
                interaction["quantity"] = random.randint(1, 3)
                # Update interaction value to be meaningful
                interaction["interaction_value"] = 5
                
            elif interaction_type == "cart":
                interaction["interaction_value"] = 3
                
            elif interaction_type == "view":
                # View duration in seconds
                interaction["view_duration"] = random.randint(5, 300)
                # Views have base value
                interaction["interaction_value"] = 1
                
            elif interaction_type == "save":
                interaction["interaction_value"] = 2
                
            elif interaction_type == "review":
                interaction["rating"] = random.randint(1, 5)
                interaction["interaction_value"] = 4
            
            interactions.append(interaction)
    
    return pd.DataFrame(interactions)

# Function to generate all datasets
def generate_synthetic_data(num_users=100, num_products=200):
    users_df = generate_users(num_users)
    products_df = generate_products(num_products)
    interactions_df = generate_interactions(users_df, products_df)
    
    return users_df, products_df, interactions_df

# Function to build user profiles from interactions
def build_user_profiles(users_df, products_df, interactions_df):
    # Create a pivot table for user-item interactions
    user_item_matrix = interactions_df.pivot_table(
        index='user_id', 
        columns='product_id', 
        values='interaction_value',
        aggfunc='sum',
        fill_value=0
    )
    
    # Create user profiles by product categories
    category_profiles = {}
    
    for user_id in user_item_matrix.index:
        category_profiles[user_id] = {}
        
        for product_id in user_item_matrix.columns:
            interaction_value = user_item_matrix.loc[user_id, product_id]
            
            if interaction_value > 0:
                # Get product category
                product = products_df[products_df['product_id'] == product_id]
                if not product.empty:
                    category = product['category'].values[0]
                    
                    # Add to category profile
                    if category in category_profiles[user_id]:
                        category_profiles[user_id][category] += interaction_value
                    else:
                        category_profiles[user_id][category] = interaction_value
    
    # Convert to DataFrame
    user_profiles = []
    for user_id, categories in category_profiles.items():
        profile = {'user_id': user_id}
        profile.update(categories)
        user_profiles.append(profile)
    
    profiles_df = pd.DataFrame(user_profiles)
    
    # Fill NaN with 0
    for category in products_df['category'].unique():
        if category not in profiles_df.columns:
            profiles_df[category] = 0
    
    profiles_df = profiles_df.fillna(0)
    
    return profiles_df, user_item_matrix

# Function to get content-based recommendations
def get_content_based_recommendations(user_id, user_profiles, products_df, n=5):
    if user_id not in user_profiles['user_id'].values:
        return pd.DataFrame()
    
    # Get user profile
    user_profile = user_profiles[user_profiles['user_id'] == user_id].iloc[0].drop('user_id')
    
    # Calculate similarity scores
    scores = []
    for _, product in products_df.iterrows():
        # Base score
        score = 0
        
        # Add points based on category preference
        if product['category'] in user_profile.index:
            category_pref = user_profile[product['category']]
            score += category_pref * 2  # Weight category preference heavily
        
        # Add points for high rating
        if product['avg_rating'] >= 4.0:
            score += product['avg_rating'] - 3  # Bonus for well-rated products
        
        # Add points for trendiness if the user likes trending items
        score += product['trending'] / 10
        
        # Add points for popular items
        score += product['popular'] / 10
        
        # Record score
        scores.append({
            'product_id': product['product_id'],
            'score': score
        })
    
    # Convert to DataFrame and sort
    scores_df = pd.DataFrame(scores)
    recommendations = scores_df.sort_values('score', ascending=False).head(n)
    
    # Merge with product details
    recommendations = recommendations.merge(products_df, on='product_id')
    
    return recommendations

# Function to get collaborative filtering recommendations
def get_collaborative_recommendations(user_id, user_item_matrix, products_df, n=5):
    if user_id not in user_item_matrix.index:
        return pd.DataFrame()
    
    # Calculate cosine similarity between users
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity, 
        index=user_item_matrix.index, 
        columns=user_item_matrix.index
    )
    
    # Find similar users
    target_user_idx = user_item_matrix.index.get_loc(user_id)
    similar_users = user_similarity_df.iloc[target_user_idx].sort_values(ascending=False)[1:6]  # Top 5 similar users
    
    # Get products that similar users interacted with
    recommendations = {}
    
    for similar_user_id, similarity in similar_users.items():
        similar_user_interactions = user_item_matrix.loc[similar_user_id]
        
        # Filter products the target user hasn't interacted with
        user_products = user_item_matrix.loc[user_id]
        new_products = similar_user_interactions[user_products == 0]
        
        # Add to recommendations with weighted scores
        for product_id, interaction_value in new_products.items():
            if interaction_value > 0:
                if product_id in recommendations:
                    recommendations[product_id] += interaction_value * similarity
                else:
                    recommendations[product_id] = interaction_value * similarity
    
    # Convert to DataFrame
    if recommendations:
        rec_df = pd.DataFrame([
            {'product_id': pid, 'score': score} 
            for pid, score in recommendations.items()
        ])
        rec_df = rec_df.sort_values('score', ascending=False).head(n)
        
        # Merge with product details
        rec_df = rec_df.merge(products_df, on='product_id')
        return rec_df
    else:
        return pd.DataFrame()

# Function to get hybrid recommendations
def get_hybrid_recommendations(user_id, user_profiles, user_item_matrix, products_df, 
                              content_weight=0.4, collab_weight=0.6, n=10):
    # Get recommendations from both methods
    content_recs = get_content_based_recommendations(user_id, user_profiles, products_df, n=n*2)
    collab_recs = get_collaborative_recommendations(user_id, user_item_matrix, products_df, n=n*2)
    
    # Combine scores
    all_recs = {}
    
    # Add content-based scores
    for _, row in content_recs.iterrows():
        all_recs[row['product_id']] = {
            'product_id': row['product_id'],
            'content_score': row['score'],
            'collab_score': 0,
            'details': {k: v for k, v in row.items() if k not in ['product_id', 'score']}
        }
    
    # Add collaborative scores
    for _, row in collab_recs.iterrows():
        if row['product_id'] in all_recs:
            all_recs[row['product_id']]['collab_score'] = row['score']
        else:
            all_recs[row['product_id']] = {
                'product_id': row['product_id'],
                'content_score': 0,
                'collab_score': row['score'],
                'details': {k: v for k, v in row.items() if k not in ['product_id', 'score']}
            }
    
    # Calculate hybrid scores
    recommendations = []
    for pid, rec in all_recs.items():
        # Normalize scores (if there are any scores to normalize)
        max_content = content_recs['score'].max() if not content_recs.empty else 1
        max_collab = collab_recs['score'].max() if not collab_recs.empty else 1
        
        norm_content = rec['content_score'] / max_content if max_content > 0 else 0
        norm_collab = rec['collab_score'] / max_collab if max_collab > 0 else 0
        
        # Calculate hybrid score
        hybrid_score = (norm_content * content_weight) + (norm_collab * collab_weight)
        
        recommendations.append({
            'product_id': pid,
            'name': rec['details']['name'],
            'category': rec['details']['category'],
            'price': rec['details']['price'],
            'avg_rating': rec['details']['avg_rating'],
            'content_score': norm_content,
            'collab_score': norm_collab,
            'hybrid_score': hybrid_score
        })
    
    # Convert to DataFrame and sort
    rec_df = pd.DataFrame(recommendations)
    if not rec_df.empty:
        rec_df = rec_df.sort_values('hybrid_score', ascending=False).head(n)
    
    return rec_df

# Main function to run the Streamlit app
def main():
    st.title("🛒 Product Recommendation System")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Dataset generation options
    st.sidebar.subheader("Dataset Generation")
    num_users = st.sidebar.slider("Number of Users", 50, 500, 100)
    num_products = st.sidebar.slider("Number of Products", 50, 500, 200)
    
    # Initialize session state for data
    if 'users_df' not in st.session_state:
        st.session_state.data_generated = False
    
    # Generate data button
    if st.sidebar.button("Fetch Data"):
        with st.spinner("Generating synthetic data..."):
            users_df, products_df, interactions_df = generate_synthetic_data(num_users, num_products)
            
            # Cache datasets in session state
            st.session_state.users_df = users_df
            st.session_state.products_df = products_df 
            st.session_state.interactions_df = interactions_df
            
            # Build user profiles
            st.session_state.user_profiles, st.session_state.user_item_matrix = build_user_profiles(
                users_df, products_df, interactions_df
            )
            
            st.session_state.data_generated = True
            st.sidebar.success("Data generated successfully!")
    
    # Main content
    if not st.session_state.get('data_generated', False):
        st.info("👈 Configure and generate data using the sidebar to start.")
        
        # Show sample UI structure even when no data is generated
        st.header("Sample Recommendation Dashboard")
        st.write("This is a preview of what the dashboard will look like after data generation.")
        
        # Create a placeholder layout
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Product Recommendations")
            st.write("Recommendations will appear here...")
            
            # Empty placeholder for recommendations
            st.markdown("#### Content-Based Recommendations")
            st.write("No data generated yet")
            
            st.markdown("#### Collaborative Filtering Recommendations")
            st.write("No data generated yet")
            
            st.markdown("#### Hybrid Recommendations")
            st.write("No data generated yet")
            
        with col2:
            st.subheader("User Profile")
            st.write("User profile details will appear here...")
            
    else:
        # Use the cached datasets
        users_df = st.session_state.users_df
        products_df = st.session_state.products_df
        interactions_df = st.session_state.interactions_df
        user_profiles = st.session_state.user_profiles
        user_item_matrix = st.session_state.user_item_matrix
        
        # Display data statistics
        st.subheader("Dataset Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Users", len(users_df))
        col2.metric("Products", len(products_df))
        col3.metric("Interactions", len(interactions_df))
        
        # User selection
        st.subheader("Select User")
        user_id = st.selectbox(
            "Select a user to generate recommendations for:",
            options=users_df['user_id'].tolist()
        )
        
        # Recommendation settings
        st.subheader("Recommendation Settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            content_weight = st.slider("Content-Based Weight", 0.0, 1.0, 0.4, 0.1)
        with col2:
            collab_weight = st.slider("Collaborative Weight", 0.0, 1.0, 0.6, 0.1)
        with col3:
            num_recommendations = st.slider("Number of Recommendations", 3, 20, 10)
        
        # Generate recommendations button
        if st.button("Generate Recommendations"):
            # Display user info
            user_info = users_df[users_df['user_id'] == user_id].iloc[0]
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.subheader("User Profile")
                st.write(f"**User ID:** {user_id}")
                st.write(f"**Age Group:** {user_info['age_group']}")
                st.write(f"**Gender:** {user_info['gender']}")
                st.write(f"**Location:** {user_info['location']}")
                
                # Display category preferences
                category_prefs = {k.replace('_pref', ''): v for k, v in user_info.items() if k.endswith('_pref')}
                
                st.write("**Category Preferences:**")
                
                # Create a radar chart of category preferences
                categories = list(category_prefs.keys())
                values = list(category_prefs.values())
                
                # Radar chart
                fig, ax = plt.subplots(figsize=(4, 3), subplot_kw=dict(polar=True))
                
                # Number of variables
                N = len(categories)
                
                # Compute angle for each axis
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Close the loop
                
                # Add values
                values += values[:1]  # Close the loop
                
                # Draw the chart
                ax.plot(angles, values, linewidth=1, linestyle='solid')
                ax.fill(angles, values, alpha=0.1)
                
                # Add category labels
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories, size=8)
                
                # Remove radial labels
                ax.set_yticklabels([])
                
                # Add title
                plt.title('Category Preferences', size=11)
                
                st.pyplot(fig)
                
                # User activity
                st.write(f"**Days Active:** {user_info['days_active']}")
                st.write(f"**Purchases per Day:** {user_info['purchase_frequency']}")
                
                # Show other metrics
                st.write("**User Traits:**")
                traits = {
                    "Price Sensitivity": user_info['price_sensitivity'],
                    "Brand Loyalty": user_info['brand_loyalty'],
                    "Trend Follower": user_info['trend_follower']
                }
                
                # Create horizontal bar chart for traits
                fig, ax = plt.subplots(figsize=(4, 2))
                bars = ax.barh(list(traits.keys()), list(traits.values()), color='skyblue')
                ax.set_xlim(0, 10)
                ax.set_xticks(range(0, 11, 2))
                
                # Add the values on the bars
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width}', 
                            ha='center', va='center', fontsize=8)
                
                st.pyplot(fig)
            
            with col1:
                st.subheader("Product Recommendations")
                
                # Get recommendations
                # Get recommendations
                with st.spinner("Generating recommendations..."):
                    # Get hybrid recommendations
                    hybrid_recs = get_hybrid_recommendations(
                        user_id, 
                        user_profiles, 
                        user_item_matrix, 
                        products_df,
                        content_weight=content_weight,
                        collab_weight=collab_weight,
                        n=num_recommendations
                    )
                    
                    # Get individual recommendations for comparison
                    content_recs = get_content_based_recommendations(
                        user_id, 
                        user_profiles, 
                        products_df, 
                        n=5
                    )
                    
                    collab_recs = get_collaborative_recommendations(
                        user_id, 
                        user_item_matrix, 
                        products_df, 
                        n=5
                    )
                
                # Display hybrid recommendations
                st.markdown("### Hybrid Recommendations")
                if not hybrid_recs.empty:
                    # Create a color mapper for scores
                    def score_to_color(score):
                        return f'background-color: rgba(0, 100, 255, {score:.2f})'
                    
                    # Format the dataframe
                    display_cols = ['name', 'category', 'price', 'avg_rating', 'hybrid_score']
                    styled_recs = hybrid_recs[display_cols].style.applymap(
                        score_to_color, 
                        subset=['hybrid_score']
                    )
                    
                    # Display the recommendations
                    st.dataframe(styled_recs)
                    
                    # Visualize recommendations by category
                    st.markdown("#### Recommendations by Category")
                    category_counts = hybrid_recs['category'].value_counts()
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    # Add count labels on bars
                    for i, count in enumerate(category_counts.values):
                        plt.text(i, count + 0.1, str(count), ha='center')
                    
                    st.pyplot(fig)
                    
                    # Price distribution of recommendations
                   
                else:
                    st.warning("No hybrid recommendations found for this user.")
                
                # Compare recommendation methods
                st.markdown("### Recommendation Method Comparison")
                
                # Create tabs for different recommendation methods
                tab1, tab2, tab3 = st.tabs(["Content-Based", "Collaborative Filtering", "Score Comparison"])
                
                with tab1:
                    if not content_recs.empty:
                        st.dataframe(content_recs[['name', 'category', 'price', 'avg_rating', 'score']])
                    else:
                        st.warning("No content-based recommendations found.")
                
                with tab2:
                    if not collab_recs.empty:
                        st.dataframe(collab_recs[['name', 'category', 'price', 'avg_rating', 'score']])
                    else:
                        st.warning("No collaborative filtering recommendations found.")
                
                with tab3:
                    # Create a comparison of recommendation scores
                    st.write("Contribution of each method to the final recommendations:")
                    
                    if not hybrid_recs.empty:
                        # Prepare score comparison data
                        score_data = hybrid_recs[['name', 'content_score', 'collab_score']].head(5)
                        
                        # Create a stacked bar chart
                        fig, ax = plt.subplots(figsize=(8, 4))
                        score_data.set_index('name').plot(kind='barh', stacked=True, ax=ax)
                        plt.xlabel('Normalized Score')
                        plt.title('Recommendation Score Composition')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("No hybrid recommendations to compare.")
        
        # Add data exploration section
        st.header("Data Exploration")
        
        # Create tabs for exploring different datasets
        tab1, tab2, tab3 = st.tabs(["Users", "Products", "Interactions"])
        
        with tab1:
            st.subheader("User Demographics")
            
            # Gender distribution
            col1, col2 = st.columns(2)
            
            with col1:
                gender_counts = users_df['gender'].value_counts()
                fig, ax = plt.subplots()
                plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
                plt.title('Gender Distribution')
                st.pyplot(fig)
            
            with col2:
                age_counts = users_df['age_group'].value_counts()
                fig, ax = plt.subplots()
                sns.barplot(x=age_counts.index, y=age_counts.values, ax=ax)
                plt.title('Age Group Distribution')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            # Location distribution
            location_counts = users_df['location'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=location_counts.index, y=location_counts.values, ax=ax)
            plt.title('User Location Distribution')
            st.pyplot(fig)
            
            # User activity summary
            st.subheader("User Activity Summary")
            
            activity_metrics = users_df[['visit_frequency', 'purchase_frequency', 'avg_session_duration', 'avg_order_value']]
            st.dataframe(activity_metrics.describe())
            
            # Correlation heatmap
          
            
            traits_cols = ['price_sensitivity', 'brand_loyalty', 'trend_follower']
            corr = users_df[traits_cols].corr()
            
            fig, ax = plt.subplots(figsize=(8, 6))
         
        
        with tab2:
            st.subheader("Product Overview")
            
            # Category distribution
            category_counts = products_df['category'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
            plt.title('Products by Category')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Add count labels on bars
            for i, count in enumerate(category_counts.values):
                plt.text(i, count + 0.5, str(count), ha='center')
            
            st.pyplot(fig)
            
            # Price distribution
            st.subheader("Price Distribution")
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x='category', y='price', data=products_df, ax=ax)
            plt.title('Price Distribution by Category')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Sales and inventory
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Products on Sale")
                sale_percent = (products_df['on_sale'].sum() / len(products_df)) * 100
                fig, ax = plt.subplots()
                plt.pie([sale_percent, 100-sale_percent], labels=['On Sale', 'Regular Price'],
                       autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
                plt.title('Percentage of Products on Sale')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Inventory Status")
                inventory_counts = products_df['inventory_status'].value_counts()
                fig, ax = plt.subplots()
                plt.pie(inventory_counts, labels=inventory_counts.index, autopct='%1.1f%%',
                       startangle=90, colors=['#99ff99','#ffcc99', '#ff9999'])
                plt.title('Inventory Status Distribution')
                st.pyplot(fig)
            
            # Product ratings
            st.subheader("Product Ratings")
            
            fig, ax = plt.subplots(figsize=(8, 5))
            # sns.histplot(products_df['avg_rating'], bins=10, kde=True, ax=ax)
            # plt.title('Distribution of Product Ratings')
            # st.pyplot(fig)
            
            # Top brands
            st.subheader("Top Brands")
            
            top_brands = products_df['brand'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=top_brands.index, y=top_brands.values, ax=ax)
            plt.title('Top 10 Brands by Product Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab3:
            st.subheader("Interaction Analysis")
            
            # Interaction types
            interaction_counts = interactions_df['interaction_type'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=interaction_counts.index, y=interaction_counts.values, ax=ax)
            plt.title('Interaction Types')
            
            # Add count labels on bars
            for i, count in enumerate(interaction_counts.values):
                plt.text(i, count + 0.5, str(count), ha='center')
                
            st.pyplot(fig)
            
            # Interaction timeline
            st.subheader("Interaction Timeline")
            
            # Convert timestamp to datetime
            interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
            interactions_df['date'] = interactions_df['timestamp'].dt.date
            
            # Group by date and interaction type
            timeline_data = interactions_df.groupby(['date', 'interaction_type']).size().unstack().fillna(0)
            
            # Plot timeline
            fig, ax = plt.subplots(figsize=(10, 6))
            timeline_data.plot(ax=ax)
            plt.title('Interactions Over Time')
            plt.xlabel('Date')
            plt.ylabel('Number of Interactions')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Category popularity
            st.subheader("Category Popularity")
            
            # Merge with products to get categories
            category_interactions = interactions_df.merge(
                products_df[['product_id', 'category']], 
                on='product_id'
            )
            
            # Count interactions by category
            category_popularity = category_interactions.groupby('category').size().sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=category_popularity.index, y=category_popularity.values, ax=ax)
            plt.title('Interactions by Product Category')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Purchase analysis
            st.subheader("Purchase Analysis")
            
            purchases = interactions_df[interactions_df['interaction_type'] == 'purchase']
            
            if not purchases.empty and 'price_paid' in purchases.columns:
                # Distribution of purchase amounts
                fig, ax = plt.subplots(figsize=(8, 5))
                # sns.histplot(purchases['price_paid'], bins=10, kde=True, ax=ax)
                # plt.title('Distribution of Purchase Amounts')
                # st.pyplot(fig)
                
                # Total purchase amount by user
                user_purchases = purchases.groupby('user_id')['price_paid'].sum().reset_index()
                user_purchases = user_purchases.sort_values('price_paid', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(x='user_id', y='price_paid', data=user_purchases, ax=ax)
                plt.title('Top 10 Users by Total Purchase Amount')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("No purchase data available for analysis.")

# Run the app
if __name__ == "__main__":
    main()