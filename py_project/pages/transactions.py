import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Initialize session state for model persistence
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'trained_scaler' not in st.session_state:
    st.session_state.trained_scaler = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

def load_and_preprocess_data(file_path):
    """
    Load CSV data and perform basic preprocessing
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Check for missing values
    st.write("Missing values:", df.isnull().sum())
    
    # Fill missing values if any
    # For numerical columns: fill with median
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # For categorical columns: fill with most frequent value
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def explore_data(df):
    """
    Perform basic exploratory data analysis
    """
    # Basic statistics
    st.write("Data shape:", df.shape)
    st.write("Data types:", df.dtypes)
    
    with st.expander("Descriptive Statistics"):
        st.dataframe(df.describe())
    
    # Fraud distribution
    st.subheader("Fraud Distribution")
    fraud_counts = df['fraud'].value_counts()
    fraud_percentage = fraud_counts / len(df) * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(fraud_counts)
    with col2:
        st.write(f"Percentage of fraudulent transactions: {fraud_percentage[1]:.2f}%")
    
    # Plot fraud distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['Legitimate', 'Fraudulent'], fraud_counts.values, color=['#3498db', '#e74c3c'])
    ax.set_title('Distribution of Transactions')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    
    return fraud_percentage[1]

def engineer_features(df):
    """
    Create additional features that might help in fraud detection
    """
    # Convert boolean columns to numeric if they aren't already
    boolean_cols = ['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']
    for col in boolean_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0})
    
    # Feature: Combination of chip and pin usage
    if 'used_chip' in df.columns and 'used_pin_number' in df.columns:
        # Convert to integer type before applying the logical AND operation
        chip = df['used_chip'].astype(int)
        pin = df['used_pin_number'].astype(int)
        df['chip_and_pin'] = (chip & pin).astype(int)
    
    # Feature: High value transaction (relative to median)
    if 'ratio_to_median_purchase_price' in df.columns:
        df['high_value_transaction'] = (df['ratio_to_median_purchase_price'] > 2.0).astype(int)
    
    return df

def prepare_features_target(df):
    """
    Prepare features and target variable for model training
    """
    # Select relevant features
    feature_cols = [col for col in df.columns if col != 'fraud']
    
    # Remove any non-numeric columns that we haven't converted
    X = df[feature_cols].select_dtypes(include=['int64', 'float64'])
    y = df['fraud']
    
    st.write("Features used:", X.columns.tolist())
    
    return X, y, X.columns.tolist()

def train_model(X, y, fraud_percentage):
    """
    Train a Random Forest model with handling for class imbalance
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Set class weights to handle imbalance
    # If fraud rate is less than 10%, use balanced weights
    if fraud_percentage < 10:
        class_weight = 'balanced'
    else:
        class_weight = None
    
    # Create a progress bar for model training
    progress_bar = st.progress(0)
    st.info("Training Random Forest model...")
    
    # Train a Random Forest model
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    progress_bar.progress(100)
    
    # Basic evaluation
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Display evaluation metrics
    st.subheader("Model Evaluation")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        st.write("Classification Report:")
        st.code(classification_report(y_test, y_pred))
    
    with col2:
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        ax.set_xticklabels(['Legitimate', 'Fraudulent'])
        ax.set_yticklabels(['Legitimate', 'Fraudulent'])
        st.pyplot(fig)
    
    # Feature importance
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    st.subheader("Feature Importance")
    st.dataframe(feature_importances)
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importances.head(10), ax=ax)
    ax.set_title('Top 10 Feature Importance')
    st.pyplot(fig)
    
    return model, scaler, X.columns.tolist()

def initialize_model(file_path):
    """
    Initialize the model by training it once
    """
    if not st.session_state.model_trained:
        st.write("Training model for the first time...")
        # Load and preprocess data
        df = load_and_preprocess_data(file_path)
        
        # Explore data
        fraud_percentage = explore_data(df)
        
        # Engineer features
        df = engineer_features(df)
        
        # Prepare features and target
        X, y, feature_names = prepare_features_target(df)
        
        # Train model
        model, scaler, feature_names = train_model(X, y, fraud_percentage)
        
        # Store model in session state
        st.session_state.trained_model = model
        st.session_state.trained_scaler = scaler
        st.session_state.feature_names = feature_names
        st.session_state.model_trained = True
        
        st.success("Model initialized successfully!")
    else:
        st.info("Model already trained. Using existing model.")

def predict_fraud(transaction_data):
    """
    Predict if a transaction is fraudulent using the pre-trained model
    
    Parameters:
    transaction_data: dict with transaction features
    
    Returns:
    dict with prediction results
    """
    # Check if model is trained
    if not st.session_state.model_trained:
        raise ValueError("Model not initialized. Call initialize_model() first.")
    
    # Convert dict to DataFrame
    transaction_df = pd.DataFrame([transaction_data])
    
    # Engineer features for the transaction
    transaction_df = engineer_features(transaction_df)
    
    # Ensure all required features are present
    for feature in st.session_state.feature_names:
        if feature not in transaction_df.columns:
            transaction_df[feature] = 0  # Default value
    
    # Select only the features used by the model
    X_transaction = transaction_df[st.session_state.feature_names]
    
    # Scale the features
    X_transaction_scaled = st.session_state.trained_scaler.transform(X_transaction)
    
    # Make prediction
    fraud_probability = st.session_state.trained_model.predict_proba(X_transaction_scaled)[0, 1]
    fraud_predicted = st.session_state.trained_model.predict(X_transaction_scaled)[0]
    
    # Define risk level
    if fraud_probability < 0.3:
        risk_level = "Low"
    elif fraud_probability < 0.7:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    # Prepare result
    result = {
        "fraud_probability": float(fraud_probability),
        "fraud_predicted": bool(fraud_predicted),
        "risk_level": risk_level
    }
    
    return result

# Main Streamlit app
def main():
    st.title("Credit Card Fraud Detection System")
    st.write("This application helps detect potentially fraudulent credit card transactions.")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["How it works", "Train Model", "Predict Fraud", "Batch Prediction", "About"]
    )
    
    if page == "How it works":
        st.header("Welcome to the Fraud Detection System")
        st.write("""
        This application uses machine learning to identify potentially fraudulent credit card transactions.
        
        **Features:**
        - Train a fraud detection model with your transaction data
        - Analyze individual transactions for fraud risk
        - Batch process multiple transactions at once
        - View detailed metrics and visualizations
        
        **How to use:**
        1. Go to the 'Train Model' page to upload your transaction data and train the model
        2. Use the 'Predict Fraud' page to analyze individual transactions
        3. For multiple transactions, use the 'Batch Prediction' page
        """)
        
        # Fixed deprecated parameter
        st.image("https://via.placeholder.com/800x400.png?text=Credit+Card+Fraud+Detection", use_container_width=True)
    
    elif page == "Train Model":
        st.header("Train Fraud Detection Model")
        st.write("Upload your transaction data to train the model.")
        
        uploaded_file = st.file_uploader("Choose a CSV file with transaction data", type=["csv"])
        
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            with open("temp_data.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Initialize model with the uploaded data
            if st.button("Train Model"):
                with st.spinner("Training in progress..."):
                    initialize_model("temp_data.csv")
        else:
            st.info("Please upload a CSV file to train the model.")
            
            # Option to use demo data
            if st.button("Use Demo Data"):
                st.warning("Using built-in demo data for training...")
                
                # Create a simple demo dataset if not found
                if not os.path.exists("card_transdata.csv"):
                    st.info("Creating demo dataset...")
                    # Create a simple synthetic dataset
                    np.random.seed(42)
                    n_samples = 1000
                    fraud_rate = 0.05
                    
                    # Generate synthetic data
                    data = {
                        'ratio_to_median_purchase_price': np.random.exponential(1, n_samples),
                        'repeat_retailer': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
                        'used_chip': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
                        'used_pin_number': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
                        'online_order': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
                        'fraud': np.random.choice([0, 1], n_samples, p=[1-fraud_rate, fraud_rate])
                    }
                    
                    # Create DataFrame and save to CSV
                    demo_df = pd.DataFrame(data)
                    demo_df.to_csv("card_transdata.csv", index=False)
                
                # Train using the demo data
                initialize_model("card_transdata.csv")
    
    elif page == "Predict Fraud":
        st.header("Predict Fraud for a Single Transaction")
        
        # Check if model is trained
        if not st.session_state.model_trained:
            st.warning("Model not trained. Please go to 'Train Model' page first.")
            if st.button("Train with Demo Data"):
                st.session_state.page = "Train Model"
                # Create a simple demo dataset if not found
                if not os.path.exists("card_transdata.csv"):
                    st.info("Creating demo dataset...")
                    # Create a simple synthetic dataset
                    np.random.seed(42)
                    n_samples = 1000
                    fraud_rate = 0.05
                    
                    # Generate synthetic data
                    data = {
                        'ratio_to_median_purchase_price': np.random.exponential(1, n_samples),
                        'repeat_retailer': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
                        'used_chip': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
                        'used_pin_number': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
                        'online_order': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
                        'fraud': np.random.choice([0, 1], n_samples, p=[1-fraud_rate, fraud_rate])
                    }
                    
                    # Create DataFrame and save to CSV
                    demo_df = pd.DataFrame(data)
                    demo_df.to_csv("card_transdata.csv", index=False)
                
                # Train using the demo data
                initialize_model("card_transdata.csv")
            return
        
        st.write("Enter transaction details to analyze for fraud risk:")
        
        # Input form for transaction details
        col1, col2 = st.columns(2)
        
        with col1:
            ratio = st.number_input("Transaction to Median Purchase Price", min_value=0.0, value=1.0, step=0.1)
            repeat_retailer = st.checkbox("Repeat Retailer", value=False)
            used_chip = st.checkbox("Used Chip", value=True)
        
        with col2:
            used_pin = st.checkbox("Used PIN", value=True)
            online_order = st.checkbox("Online Order", value=False)
        
        # Create transaction data dictionary
        transaction = {
            'ratio_to_median_purchase_price': ratio,
            'repeat_retailer': repeat_retailer,
            'used_chip': used_chip,
            'used_pin_number': used_pin,
            'online_order': online_order
        }
        
        # Predict button
        if st.button("Analyze Transaction"):
            with st.spinner("Analyzing..."):
                result = predict_fraud(transaction)
                
                # Display results
                st.subheader("Fraud Detection Results")
                
                # Use columns for layout
                col1, col2, col3 = st.columns(3)
                
                # Color based on risk level
                if result["risk_level"] == "Low":
                    color = "green"
                elif result["risk_level"] == "Medium":
                    color = "orange"
                else:
                    color = "red"
                
                with col1:
                    st.metric("Fraud Probability", f"{result['fraud_probability']:.2%}")
                
                with col2:
                    st.metric("Prediction", "Fraudulent" if result['fraud_predicted'] else "Legitimate")
                
                with col3:
                    st.markdown(f"<h3 style='color:{color};'>Risk: {result['risk_level']}</h3>", unsafe_allow_html=True)
                
                # FIXED: Improved gauge chart for fraud probability visualization
                fig, ax = plt.subplots(figsize=(8, 3))
                
                # Create a proper scale for the gauge
                # Draw a background bar (grey)
                ax.barh(0, 1, color='lightgray', height=0.5, alpha=0.3)
                
                # Now draw the actual probability bar with proper color
                fraud_prob = result['fraud_probability']
                ax.barh(0, fraud_prob, color=color, height=0.5)
                
                # Set proper limits and labels
                ax.set_xlim(0, 1)
                ax.set_ylim(-1, 1)
                ax.set_yticks([])
                ax.set_xticks([0, 0.3, 0.7, 1])
                ax.set_xticklabels(['0%', '30%', '70%', '100%'])
                ax.set_xlabel('Fraud Probability')
                
                # Add risk markers with improved visibility
                ax.axvline(x=0.3, color='green', linestyle='--', alpha=0.7, linewidth=2)
                ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, linewidth=2)
                
                # Add text labels for risk zones with better positioning
                ax.text(0.15, -0.7, 'Low Risk', ha='center', va='center', color='green', fontweight='bold')
                ax.text(0.5, -0.7, 'Medium Risk', ha='center', va='center', color='orange', fontweight='bold')
                ax.text(0.85, -0.7, 'High Risk', ha='center', va='center', color='red', fontweight='bold')
                
                # Add the actual percentage value on top of the bar
                ax.text(min(max(fraud_prob, 0.05), 0.95), 0, 
                        f"{fraud_prob:.1%}", 
                        ha='center', va='center', 
                        color='white' if fraud_prob > 0.3 else 'black',
                        fontweight='bold')
                
                # Improve overall appearance
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                # Display the chart
                st.pyplot(fig)
                
                # Explanation
                st.subheader("Analysis Explanation")
                explanation = []
                
                if ratio > 2.0:
                    explanation.append("- Transaction amount is significantly higher than the median purchase price.")
                
                if not repeat_retailer:
                    explanation.append("- Transaction is with a new retailer.")
                
                if not used_chip and not used_pin:
                    explanation.append("- Transaction did not use chip or PIN security features.")
                
                if online_order:
                    explanation.append("- Online orders typically have higher fraud risk.")
                
                if not explanation:
                    explanation.append("- No significant risk factors identified.")
                
                for line in explanation:
                    st.write(line)
    
    elif page == "Batch Prediction":
        st.header("Batch Fraud Prediction")
        
        # Check if model is trained
        if not st.session_state.model_trained:
            st.warning("Model not trained. Please go to 'Train Model' page first.")
            if st.button("Train with Demo Data"):
                st.session_state.page = "Train Model"
                # Create a simple demo dataset if not found
                if not os.path.exists("card_transdata.csv"):
                    st.info("Creating demo dataset...")
                    # Create a simple synthetic dataset
                    np.random.seed(42)
                    n_samples = 1000
                    fraud_rate = 0.05
                    
                    # Generate synthetic data
                    data = {
                        'ratio_to_median_purchase_price': np.random.exponential(1, n_samples),
                        'repeat_retailer': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
                        'used_chip': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
                        'used_pin_number': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
                        'online_order': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
                        'fraud': np.random.choice([0, 1], n_samples, p=[1-fraud_rate, fraud_rate])
                    }
                    
                    # Create DataFrame and save to CSV
                    demo_df = pd.DataFrame(data)
                    demo_df.to_csv("card_transdata.csv", index=False)
                
                # Train using the demo data
                initialize_model("card_transdata.csv")
            return
        
        st.write("Upload a CSV file with multiple transactions to analyze.")
        
        # Example template
        st.write("Your CSV should contain these columns:")
        example_df = pd.DataFrame({
            'ratio_to_median_purchase_price': [1.5, 2.7],
            'repeat_retailer': [True, False],
            'used_chip': [True, False],
            'used_pin_number': [True, False],
            'online_order': [False, True]
        })
        
        st.dataframe(example_df)
        
        # Download template button
        template_csv = example_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV Template",
            data=template_csv,
            file_name="transaction_template.csv",
            mime="text/csv"
        )
        
        # Upload batch file
        batch_file = st.file_uploader("Upload batch transactions CSV", type=["csv"])
        
        if batch_file is not None:
            # Read the batch data
            batch_df = pd.read_csv(batch_file)
            
            st.write("Preview of uploaded data:")
            st.dataframe(batch_df.head())
            
            if st.button("Analyze Batch"):
                with st.spinner("Analyzing transactions..."):
                    # Process each transaction
                    results = []
                    for i, row in batch_df.iterrows():
                        # Convert row to dict
                        transaction = row.to_dict()
                        # Predict
                        result = predict_fraud(transaction)
                        # Add to results
                        results.append({
                            "Transaction": i+1,
                            "Fraud Probability": f"{result['fraud_probability']:.2%}",
                            "Prediction": "Fraudulent" if result['fraud_predicted'] else "Legitimate",
                            "Risk Level": result['risk_level']
                        })
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame(results)
                    
                    # Show results
                    st.subheader("Batch Analysis Results")
                    st.dataframe(results_df)
                    
                    # Summary statistics
                    fraudulent_count = sum(1 for r in results if r["Prediction"] == "Fraudulent")
                    high_risk_count = sum(1 for r in results if r["Risk Level"] == "High")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Transactions", len(results))
                    with col2:
                        st.metric("Fraudulent Transactions", fraudulent_count)
                    with col3:
                        st.metric("High Risk Transactions", high_risk_count)
                    
                    # Download results button
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results CSV",
                        data=csv,
                        file_name="fraud_analysis_results.csv",
                        mime="text/csv"
                    )
    
    elif page == "About":
        st.header("About This Application")
        st.write("""
        ## Credit Card Fraud Detection System
        
        This application uses machine learning to detect potentially fraudulent credit card transactions. 
        It employs a Random Forest classifier trained on historical transaction data to identify patterns 
        and signatures of fraudulent activities.
        
        ### Key Features:
        - Real-time transaction analysis
        - Batch processing capabilities
        - Detailed risk assessment
        - User-friendly interface
        
        ### How It Works:
        The model analyzes various transaction features such as:
        - Transaction amount relative to typical purchases
        - Whether the retailer has been visited before
        - Security features used (chip/PIN)
        - Whether the transaction was online
        
        Based on these factors, it calculates a fraud probability and risk level.
        
        ### About the Model:
        The underlying model is a Random Forest classifier that has been optimized to handle 
        imbalanced datasets, which is common in fraud detection where most transactions are legitimate.
        """)
        
        st.write("### Developed by: Your Name")
        st.write("Version 1.0")

if __name__ == "__main__":
    main()