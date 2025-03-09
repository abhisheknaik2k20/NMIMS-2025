import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import pickle
import os
from geopy.distance import geodesic
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import threading
from geopy.distance import geodesic
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class CRMAnomalyDetector:
    """
    A comprehensive anomaly detection system for CRM data that can identify:
    1. Unusual purchase patterns
    2. Suspicious login locations
    """
    
    def __init__(self):
        # Models for different anomaly types
        self.purchase_model = None
        self.location_model = None
        
        # Scalers for preprocessing
        self.purchase_scaler = StandardScaler()
        
        # Known user locations (user_id -> list of [lat, long] coordinates)
        self.user_locations = {}
        
        # User purchase history stats (user_id -> dict of stats)
        self.user_purchase_stats = {}
    
    def fit_purchase_model(self, purchase_data):
        """
        Train a model to detect unusual purchase patterns.
        
        Parameters:
        purchase_data (pd.DataFrame): DataFrame with columns:
            - user_id: Unique identifier for the user
            - amount: Purchase amount
            - category: Product category
            - timestamp: Time of purchase
            - frequency: How often user makes purchases (optional)
        """
        # Extract relevant features
        features = self._extract_purchase_features(purchase_data)
        
        # Calculate per-user stats for future comparison
        self._calculate_user_purchase_stats(purchase_data)
        
        # Scale features
        scaled_features = self.purchase_scaler.fit_transform(features)
        
        # Train Isolation Forest model
        self.purchase_model = IsolationForest(contamination=0.05, random_state=42)
        self.purchase_model.fit(scaled_features)
        
        return self
    
    def _extract_purchase_features(self, purchase_data):
        """Extract features from purchase data for anomaly detection."""
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(purchase_data['timestamp']):
            purchase_data['timestamp'] = pd.to_datetime(purchase_data['timestamp'])
        
        # Group by user and calculate features
        user_features = []
        
        for user_id, user_data in purchase_data.groupby('user_id'):
            # Calculate time between purchases
            user_data = user_data.sort_values('timestamp')
            time_diffs = user_data['timestamp'].diff().dt.total_seconds() / 3600  # in hours
            
            # Basic statistics
            avg_amount = user_data['amount'].mean()
            max_amount = user_data['amount'].max()
            std_amount = user_data['amount'].std() if len(user_data) > 1 else 0
            avg_time_between = time_diffs.mean() if len(time_diffs) > 1 else 24  # default to 24h
            
            # Category diversity (number of unique categories / total purchases)
            category_diversity = len(user_data['category'].unique()) / len(user_data)
            
            # Purchase frequency (purchases per day)
            days_span = (user_data['timestamp'].max() - user_data['timestamp'].min()).total_seconds() / 86400
            days_span = max(days_span, 1)  # Avoid division by zero
            purchase_frequency = len(user_data) / days_span
            
            user_features.append([
                avg_amount, max_amount, std_amount, 
                avg_time_between, category_diversity, purchase_frequency
            ])
        
        return pd.DataFrame(user_features, columns=[
            'avg_amount', 'max_amount', 'std_amount', 
            'avg_time_between', 'category_diversity', 'purchase_frequency'
        ])
    
    def _calculate_user_purchase_stats(self, purchase_data):
        """Calculate comprehensive per-user purchase statistics for real-time detection."""
        for user_id, user_data in purchase_data.groupby('user_id'):
            # Convert timestamp to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(user_data['timestamp']):
                user_data['timestamp'] = pd.to_datetime(user_data['timestamp'])
                
            user_data = user_data.sort_values('timestamp')
            
            # Calculate basic stats for this user
            user_stats = {
                'avg_amount': user_data['amount'].mean(),
                'median_amount': user_data['amount'].median(),
                'max_amount': user_data['amount'].max(),
                'min_amount': user_data['amount'].min(),
                'std_amount': user_data['amount'].std() if len(user_data) > 1 else 0,
                'categories': user_data['category'].value_counts().to_dict(),
                'total_purchases': len(user_data),
                'first_purchase_date': user_data['timestamp'].min(),
                'last_purchase_date': user_data['timestamp'].max()
            }
            
            # Time pattern analysis
            user_stats['purchase_times'] = [ts.hour for ts in user_data['timestamp']]
            user_stats['purchase_days'] = [ts.dayofweek for ts in user_data['timestamp']]
            
            # Calculate time intervals between purchases
            time_diffs = user_data['timestamp'].diff().dt.total_seconds() / 60  # minutes
            user_stats['avg_time_between_purchases'] = time_diffs.mean() if not pd.isna(time_diffs.mean()) else 1440  # default to 24h
            user_stats['min_time_between_purchases'] = time_diffs.min() if not pd.isna(time_diffs.min()) else 1440
            
            # Purchase frequency statistics
            days_active = (user_data['timestamp'].max() - user_data['timestamp'].min()).total_seconds() / 86400
            days_active = max(days_active, 1)  # Avoid division by zero
            user_stats['purchases_per_day'] = len(user_data) / days_active
            
            # Store known devices and IPs if available
            if 'device_id' in user_data.columns:
                user_stats['known_devices'] = user_data['device_id'].unique().tolist()
            
            if 'ip_address' in user_data.columns:
                user_stats['known_ips'] = user_data['ip_address'].unique().tolist()
            
            # Store purchase locations if available
            if 'latitude' in user_data.columns and 'longitude' in user_data.columns:
                user_stats['purchase_locations'] = user_data[['latitude', 'longitude']].values.tolist()
            
            # Calculate category transition probabilities (for sequence anomalies)
            if len(user_data) > 1:
                category_transitions = {}
                prev_category = None
                
                for category in user_data['category']:
                    if prev_category is not None:
                        transition_key = f"{prev_category}->{category}"
                        if transition_key not in category_transitions:
                            category_transitions[transition_key] = 0
                        category_transitions[transition_key] += 1
                    prev_category = category
                
                user_stats['category_transitions'] = category_transitions
            
            # Store calculated stats for this user
            self.user_purchase_stats[user_id] = user_stats
    
    def fit_location_model(self, login_data):
        """
        Train a model to detect unusual login locations.
        
        Parameters:
        login_data (pd.DataFrame): DataFrame with columns:
            - user_id: Unique identifier for the user
            - latitude: Latitude of login location
            - longitude: Longitude of login location
            - timestamp: Time of login
            - device_id: Device identifier (optional)
        """
        # Store known locations for each user
        for user_id, user_data in login_data.groupby('user_id'):
            locations = user_data[['latitude', 'longitude']].values
            self.user_locations[user_id] = locations
            
        return self
    
    def detect_unusual_purchase(self, purchase):
        """
        Detect if a new purchase is anomalous with enhanced detection techniques.
        
        Parameters:
        purchase (dict): Dictionary with keys:
            - user_id: Unique identifier for the user
            - amount: Purchase amount
            - category: Product category
            - timestamp: Time of purchase
            - device_id: Device identifier (optional)
            - ip_address: IP address (optional)
        
        Returns:
        tuple: (is_anomalous, anomaly_score, reason)
        """
        user_id = purchase['user_id']
        amount = purchase['amount']
        category = purchase['category']
        timestamp = pd.to_datetime(purchase['timestamp'])
        
        # If we have no history for this user, flag as anomalous
        if user_id not in self.user_purchase_stats:
            return True, 0.8, "New user with no purchase history"
        
        user_stats = self.user_purchase_stats[user_id]
        reasons = []
        anomaly_score = 0.0
        
        # 1. Check amount against user history with a Z-score approach
        amount_z_score = 0
        if user_stats['std_amount'] > 0:
            amount_z_score = (amount - user_stats['avg_amount']) / user_stats['std_amount']
            if amount_z_score > 3:  # More than 3 standard deviations
                score_increment = min(1.0, abs(amount_z_score) / 10)
                reasons.append(f"Purchase amount (${amount:.2f}) is unusually high (z-score: {amount_z_score:.2f})")
                anomaly_score += score_increment
        elif amount > 2 * user_stats['avg_amount']:
            reasons.append(f"Purchase amount (${amount:.2f}) is more than twice the usual amount")
            anomaly_score += 0.5
        
        # 2. Check if category is unusual for this user
        if category not in user_stats['categories']:
            reasons.append(f"User has never purchased from category '{category}' before")
            anomaly_score += 0.3
        else:
            # Calculate category frequency for this user
            category_freq = user_stats['categories'][category] / user_stats['total_purchases']
            if category_freq < 0.1:  # Less than 10% of purchases
                reasons.append(f"User rarely purchases from category '{category}' ({category_freq:.1%} of purchases)")
                anomaly_score += 0.2
        
        # 3. Check time patterns with greater sophistication
        hour = timestamp.hour
        day = timestamp.dayofweek
        
        # Time of day analysis
        if hour not in user_stats['purchase_times']:
            reasons.append(f"User has never made purchases at this time of day ({hour}:00)")
            anomaly_score += 0.2
        else:
            hour_count = user_stats['purchase_times'].count(hour)
            hour_frequency = hour_count / len(user_stats['purchase_times'])
            if hour_frequency < 0.1:  # Less than 10% of purchases at this hour
                reasons.append(f"User rarely makes purchases at this time ({hour}:00) - only {hour_frequency:.1%} of the time")
                anomaly_score += 0.15
        
        # Day of week analysis
        if day not in user_stats['purchase_days']:
            reasons.append(f"User has never made purchases on this day of week ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day]})")
            anomaly_score += 0.2
        else:
            day_count = user_stats['purchase_days'].count(day)
            day_frequency = day_count / len(user_stats['purchase_days'])
            if day_frequency < 0.1:  # Less than 10% of purchases on this day
                reasons.append(f"User rarely makes purchases on {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day]} - only {day_frequency:.1%} of the time")
                anomaly_score += 0.15
        
        # 4. Purchase velocity check (multiple purchases in short timespan)
        if 'last_purchase_timestamp' in purchase:
            last_timestamp = pd.to_datetime(purchase['last_purchase_timestamp'])
            minutes_since_last = (timestamp - last_timestamp).total_seconds() / 60
            
            # Define rapid purchase threshold (e.g., 5 minutes)
            if minutes_since_last < 5:
                reasons.append(f"Rapid successive purchase ({minutes_since_last:.1f} minutes after previous purchase)")
                anomaly_score += 0.4
        
        # 5. Multi-location purchase detection (if available)
        if 'latitude' in purchase and 'longitude' in purchase and 'last_purchase_location' in purchase:
            current_location = (purchase['latitude'], purchase['longitude'])
            last_location = purchase['last_purchase_location']
            
            # Check distance between purchase locations
            if last_location:
                distance = geodesic(current_location, last_location).kilometers
                
                # Get time difference
                if 'last_purchase_timestamp' in purchase:
                    last_timestamp = pd.to_datetime(purchase['last_purchase_timestamp'])
                    hours_diff = (timestamp - last_timestamp).total_seconds() / 3600
                    
                    # Calculate feasible travel distance (assuming 100 km/h average travel speed)
                    max_feasible_distance = hours_diff * 100
                    
                    if distance > max_feasible_distance and hours_diff < 24:
                        anomaly_score += 0.7
                        reasons.append(f"Impossible purchase location: {distance:.1f} km from previous purchase location in {hours_diff:.1f} hours")
        
        # 6. Device/IP consistency check (if available)
        if 'device_id' in purchase and 'known_devices' in user_stats:
            device_id = purchase['device_id']
            if device_id not in user_stats['known_devices']:
                reasons.append(f"Purchase made from new device")
                anomaly_score += 0.3
        
        if 'ip_address' in purchase and 'known_ips' in user_stats:
            ip_address = purchase['ip_address']
            if ip_address not in user_stats['known_ips']:
                reasons.append(f"Purchase made from new IP address")
                anomaly_score += 0.25
        
        # 7. Purchase amount relative to user income (if available)
        if 'income_bracket' in user_stats and 'income_bracket' in purchase:
            if purchase['income_bracket'] < user_stats['income_bracket']:
                if amount > user_stats['max_amount']:
                    reasons.append(f"High value purchase (${amount:.2f}) unusual for income bracket")
                    anomaly_score += 0.35
        
        # Calculate final anomaly status
        # Using a threshold approach similar to login detection
        is_anomalous = anomaly_score >= 0.6
        
        return is_anomalous, min(1.0, anomaly_score), ", ".join(reasons) if reasons else "No anomalies detected"
    
    def detect_unusual_login(self, login):
        """
        Detect if a login location is anomalous.
        
        Parameters:
        login (dict): Dictionary with keys:
            - user_id: Unique identifier for the user
            - latitude: Latitude of login location
            - longitude: Longitude of login location
            - timestamp: Time of login
            - ip_address: IP address (optional)
            - device_id: Device identifier (optional)
        
        Returns:
        tuple: (is_anomalous, anomaly_score, reason)
        """
        user_id = login['user_id']
        lat = login['latitude']
        lon = login['longitude']
        current_location = (lat, lon)
        timestamp = pd.to_datetime(login['timestamp'])
        
        # If we have no history for this user, flag as anomalous
        if user_id not in self.user_locations:
            return True, 0.8, "New user with no login history"
        
        known_locations = self.user_locations[user_id]
        
        # Find minimum distance to any known location
        min_distance = float('inf')
        for loc in known_locations:
            known_loc = (loc[0], loc[1])
            distance = geodesic(current_location, known_loc).kilometers
            min_distance = min(min_distance, distance)
        
        # Define thresholds for anomaly detection
        suspicious_threshold = 500  # km
        highly_suspicious_threshold = 2000  # km
        
        # Calculate an anomaly score based on distance
        anomaly_score = 0.0
        reason = ""
        
        if min_distance > highly_suspicious_threshold:
            anomaly_score = 1.0
            reason = f"Login location is {min_distance:.1f} km from any known location for this user"
        elif min_distance > suspicious_threshold:
            anomaly_score = 0.7
            reason = f"Login location is {min_distance:.1f} km from any known location for this user"
        elif min_distance > 100:
            anomaly_score = 0.3
            reason = f"Login location is {min_distance:.1f} km from any known location for this user"
        else:
            reason = "Login location is consistent with user history"
        
        # Check for impossible travel
        if 'last_login_timestamp' in login and 'last_login_location' in login:
            last_timestamp = pd.to_datetime(login['last_login_timestamp'])
            last_location = login['last_login_location']
            
            hours_diff = (timestamp - last_timestamp).total_seconds() / 3600
            distance = geodesic(current_location, last_location).kilometers
            
            # Assume maximum travel speed of 1000 km/h (commercial flight)
            max_possible_distance = hours_diff * 1000
            
            if distance > max_possible_distance and hours_diff > 0:
                anomaly_score = max(anomaly_score, 0.9)
                reason += f", Impossible travel: {distance:.1f} km in {hours_diff:.1f} hours"
        
        is_anomalous = anomaly_score >= 0.6
        
        return is_anomalous, anomaly_score, reason
    
    def save_model(self, filepath):
        """Save the trained model to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'purchase_model': self.purchase_model,
                'purchase_scaler': self.purchase_scaler,
                'user_locations': self.user_locations,
                'user_purchase_stats': self.user_purchase_stats,
            }, f)
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls()
        model.purchase_model = data['purchase_model']
        model.purchase_scaler = data['purchase_scaler']
        model.user_locations = data['user_locations']
        model.user_purchase_stats = data['user_purchase_stats']
        
        return model

# Set page configuration
st.set_page_config(
    page_title="CRM Anomaly Detector",
    page_icon="🔍",
    layout="wide"
)

# Initialize Firebase connection
@st.cache_resource
def get_firebase_client(_cred_dict=None):
    """Initialize Firebase Admin SDK with provided credentials"""
    # Check if already initialized
    if not firebase_admin._apps:
        if _cred_dict:
            cred = credentials.Certificate(_cred_dict)
            firebase_admin.initialize_app(cred)
    
    # Return Firestore client
    return firestore.client()
    
@st.cache_resource
def initialize_firebase():
    """Initialize Firebase Admin SDK directly from file"""
    # Check if already initialized
    if not firebase_admin._apps:
        try:
            # Direct path to the credentials file
            cred = credentials.Certificate('firebase-credentials.json')
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Firebase initialization error: {str(e)}")
            st.error("Please ensure 'firebase-credentials.json' exists in the app directory")
            st.stop()   
    # Return Firestore client
    return firestore.client()
def setup_real_time_monitoring(detector):
    st.title("🔄 Real-Time Anomaly Monitoring")
    
    # Import required libraries
    from datetime import datetime
    import time
    from google.cloud import firestore
    import random  # For generating demo anomalies if needed
    
    # Select collection to monitor
    monitor_collection = st.selectbox(
        "Select collection to monitor for new entries:",
        ["logins", "purchases"]
    )
    
    # Create placeholder for real-time results and alerts
    alert_placeholder = st.empty()  # New placeholder for alerts
    status_indicator = st.empty()
    latest_results = st.container()
    
    # Initialize monitoring status and results storage in session state
    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = False
        st.session_state.detection_results = []
        st.session_state.last_check_time = None
        st.session_state.query_cursor = None
        st.session_state.auto_refresh_interval = 1  # Default refresh interval in seconds
        st.session_state.alert_shown = False  # Track if alert is currently shown
        st.session_state.alert_message = ""  # Store alert message
        st.session_state.alert_level = "info"  # Store alert level
        st.session_state.anomaly_count_total = 0  # Track total anomalies
        st.session_state.entry_count_total = 0  # Track total entries
    
    # Controls for starting/stopping monitoring and setting refresh interval
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not st.session_state.monitoring_active:
            if st.button("▶️ Start Monitoring", use_container_width=True):
                st.session_state.monitoring_active = True
                st.session_state.last_check_time = time.time()
                # Reset the cursor when starting fresh
                st.session_state.query_cursor = None
                st.session_state.last_check_time = None
                st.session_state.alert_shown = False  # Reset alert state
                st.rerun()
    with col2:
        if st.session_state.monitoring_active:
            if st.button("⏹️ Stop Monitoring", use_container_width=True):
                st.session_state.monitoring_active = False
                st.rerun()
    
    with col3:
        # Adjust refresh interval
        if st.session_state.monitoring_active:
            refresh_options = {
                "Fast (0.5s)": 0.5,
                "Normal (1s)": 1,
                "Slow (3s)": 3,
                "Very Slow (5s)": 5
            }
            selected_refresh = st.selectbox(
                "Refresh Rate:",
                list(refresh_options.keys()),
                index=1
            )
            st.session_state.auto_refresh_interval = refresh_options[selected_refresh]
    
    # Display current status with animation
    if st.session_state.monitoring_active:
        status_indicator.markdown(f"""
        <div style="display: flex; align-items: center; background-color: #e6f7e6; padding: 10px; border-radius: 5px; border-left: 5px solid #28a745;">
            <div style="animation: pulse 1.5s infinite; margin-right: 10px;">🔄</div>
            <div>
                <strong>ACTIVE MONITORING:</strong> Scanning for new entries in '<code>{monitor_collection}</code>'
                <div style="font-size: 0.8em; opacity: 0.8;">Refresh interval: {st.session_state.auto_refresh_interval}s</div>
            </div>
        </div>
        <style>
        @keyframes pulse {{
    0% {{ transform: scale(1); }}
    50% {{ transform: scale(1.2); }}
    100% {{ transform: scale(1); }}
}}
        </style>
        """, unsafe_allow_html=True)
    else:
        status_indicator.markdown("""
        <div style="display: flex; align-items: center; background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 5px solid #ffc107;">
            <div style="margin-right: 10px;">⏸️</div>
            <div>
                <strong>MONITORING PAUSED</strong>
                <div style="font-size: 0.8em; opacity: 0.8;">Click "Start Monitoring" to begin</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Function to check for new entries
    def check_for_new_entries():
        try:
            db = initialize_firebase()
            
            # Build the initial query
            query = db.collection(monitor_collection)
            
            # If we have a cursor, use it for pagination
            if st.session_state.query_cursor:
                query = query.start_after(st.session_state.query_cursor)
            
            # Order by document ID or creation time
            query = query.order_by('__name__')
            
            # Limit to a reasonable number of new records
            query = query.limit(10)
            
            # Execute query
            docs = query.get()
            new_entries_count = len(docs)
            anomaly_count = 0  # Track number of anomalies in this batch
            
            # Update the cursor for next query
            if new_entries_count > 0:
                st.session_state.query_cursor = docs[-1]
            
            # Process each new document
            for doc in docs:
                doc_data = doc.to_dict()
                doc_data['doc_id'] = doc.id
                
                # Update total entries count
                st.session_state.entry_count_total += 1
                
                # Process different types of entries
                if monitor_collection == "purchases":
                    result = detector.detect_unusual_purchase(doc_data)
                    detection_type = "Purchase Anomaly"
                else:  # logins
                    result = detector.detect_unusual_login(doc_data)
                    detection_type = "Login Anomaly"
                
                # Store the result with timestamp and raw data
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                is_anomalous, score, reason = result
                
                # Add pulse animation class for new entries
                animation_class = "new-entry-pulse" if is_anomalous else "normal-entry"
                
                st.session_state.detection_results.append({
                    "timestamp": timestamp,
                    "detection_type": detection_type,
                    "result": result,
                    "raw_data": doc_data,
                    "animation_class": animation_class,
                    "fresh": True  # Flag to indicate this is a new entry
                })
                
                # Save to anomaly_results collection if anomalous
                if is_anomalous:
                    anomaly_count += 1  # Increment anomaly count
                    st.session_state.anomaly_count_total += 1
                    try:
                        result_data = {
                            "detection_type": "purchase" if monitor_collection == "purchases" else "login",
                            "user_id": doc_data.get("user_id", "unknown"),
                            "is_anomalous": is_anomalous,
                            "anomaly_score": score,
                            "reason": reason,
                            "timestamp": timestamp,
                            f"{monitor_collection[:-1]}_data": doc_data
                        }
                        db.collection("anomaly_results").add(result_data)
                    except Exception as e:
                        st.error(f"Error saving anomaly result: {e}")
            
            # Set alert message and level based on findings
            if new_entries_count > 0:
                if anomaly_count > 0:
                    st.session_state.alert_message = f"⚠️ ALERT! {anomaly_count} anomalies detected in {new_entries_count} new entries!"
                    st.session_state.alert_level = "error"
                    st.session_state.alert_shown = True
                    # Use a sound alert for anomalies (HTML/JS solution)
                    st.markdown(
                        """
                        <script>
                        // Play alert sound for anomaly detection
                        var audio = new Audio("data:audio/wav;base64,//uQRAAAAWMSLwUIYAAsYkXgoQwAEaYLWfkWgAI0wWs/ItAAAGDgYtAgAyN+QWaAAihwMWm4G8QQRDiMcCBcH3Cc+CDv/7xA4Tvh9Rz/y8QADBwMWgQAZG/ILNAARQ4GLTcDeIIIhxGOBAuD7hOfBB3/94gcJ3w+o5/5eIAIAAAVwWgQAVQ2ORaIQwEMAJiDg95G4nQL7mQVWI6GwRcfsZAcsKkJvxgxEjzFUgfHoSQ9Qq7KNwqHwuB13MA4a1q/DmBrHgPcmjiGoh//EwC5nGPEmS4RcfkVKOhJf+WOgoxJclFz3kgn//dBA+ya1GhurNn8zb//9NNutNuhz31f////9vt///z+IdAEAAAK4LQIAKobHItEIYCGAExBwe8jcToF9zIKrEdDYIuP2MgOWFSE34wYiR5iqQPj0JIeoVdlG4VD4XA67mAcNa1fhzA1jwHuTRxDUQ//iYBczjHiTJcIuPyKlHQkv/LHQUYkuSi57yQT//uggfZNajQ3Vm38///////+7Nwz/Zq//////////+/////////+/////////////////////////////////////////+///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////8=");
                        audio.play();
                        </script>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.session_state.alert_message = f"👁️ {new_entries_count} new entries detected (no anomalies)"
                    st.session_state.alert_level = "info"
                    st.session_state.alert_shown = True
            
            # Return the count of new entries
            return new_entries_count, anomaly_count
            
        except Exception as e:
            st.error(f"Error checking for new entries: {e}")
            st.exception(e)  # This will show the full exception for debugging
            return 0, 0
    
    # Show alert if active - with more expressive visual styling
    if st.session_state.alert_shown and st.session_state.alert_message:
        # Determine alert style based on level
        if st.session_state.alert_level == "error":
            alert_placeholder.markdown(f"""
            <div style="background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; 
                        border-left: 6px solid #dc3545; animation: flash 1s 3; font-weight: bold; margin-bottom: 20px;">
                <div style="display: flex; align-items: center;">
                    <div style="font-size: 1.5em; margin-right: 10px;">⚠️</div>
                    <div>{st.session_state.alert_message}</div>
                </div>
            </div>
            <style>
                @keyframes flash {{
                    0% {{ background-color: #f8d7da; }}
                    50% {{ background-color: #ff6b6b; }}
                    100% {{ background-color: #f8d7da; }}
                }}
            </style>
            """, unsafe_allow_html=True)
        else:
            alert_placeholder.markdown(f"""
            <div style="background-color: #d1ecf1; color: #0c5460; padding: 12px; border-radius: 5px; 
                        border-left: 6px solid #17a2b8; font-weight: 500; margin-bottom: 20px;">
                <div style="display: flex; align-items: center;">
                    <div style="font-size: 1.2em; margin-right: 10px;">👁️</div>
                    <div>{st.session_state.alert_message}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Initialize the auto_refresh key in session state if not present
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    
    # Reliable auto-refresh approach using Streamlit's native functionality
    if st.session_state.monitoring_active:
        # On initial load or when monitoring starts, check for existing entries first
        if st.session_state.query_cursor is None:
            with st.spinner("Initial check for existing entries..."):
                new_entries_count, anomaly_count = check_for_new_entries()
                if new_entries_count > 0:
                    st.success(f"Found {new_entries_count} existing entries" + 
                              (f" including {anomaly_count} anomalies!" if anomaly_count > 0 else ""))
                else:
                    st.info("No existing entries found to process. Monitoring for new ones...")
        
        # Automatic periodic checking
        current_time = time.time()
        
        # Check if it's time to refresh based on the user-defined interval
        if st.session_state.last_check_time is None or (current_time - st.session_state.last_check_time >= st.session_state.auto_refresh_interval):
            st.session_state.last_check_time = current_time
            
            # Create a placeholder for the refresh status
            refresh_placeholder = st.empty()
            with refresh_placeholder:
                with st.spinner("Checking for new entries..."):
                    new_entries_count, anomaly_count = check_for_new_entries()
            
            # If we found new entries, display a message
            if new_entries_count > 0:
                if anomaly_count > 0:
                    refresh_placeholder.markdown(f"""
                    <div style="background-color: #fff3cd; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
                        <span style="color: #ff0000; font-weight: bold;">🔔 Found {new_entries_count} new entries with {anomaly_count} anomalies!</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    refresh_placeholder.markdown(f"""
                    <div style="background-color: #d4edda; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
                        <span style="color: #155724; font-weight: bold;">✓ Found {new_entries_count} new entries (all normal)</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Set up automatic rerun after the interval
            if st.session_state.monitoring_active and st.session_state.auto_refresh:
                # Calculate how long until next refresh
                seconds_to_next = st.session_state.auto_refresh_interval
                
                # Force an immediate rerun if entries were found
                if new_entries_count > 0:
                    time.sleep(0.5)  # Brief pause to let the UI update
                    st.rerun()
                    
                # Add a container for the countdown timer
                countdown_placeholder = st.empty()
                
                # Implement the countdown timer with small steps
                for i in range(int(seconds_to_next * 10)):
                    if not st.session_state.monitoring_active:
                        break
                        
                    # Update countdown text with more visual indicator
                    remaining = seconds_to_next - (i / 10)
                    progress_pct = 100 - (remaining / seconds_to_next * 100)
                    
                    countdown_placeholder.markdown(f"""
                    <div style="margin: 10px 0;">
                        <div style="display: flex; align-items: center; font-size: 0.9em; color: #666;">
                            <div>Next check in: {remaining:.1f}s</div>
                            <div style="flex-grow: 1; margin: 0 10px;">
                                <div style="background-color: #e9ecef; height: 6px; border-radius: 3px;">
                                    <div style="background-color: #4CAF50; width: {progress_pct}%; height: 6px; border-radius: 3px;"></div>
                                </div>
                            </div>
                            <div>⏱️</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Small sleep interval for a smoother countdown
                    time.sleep(0.1)
                
                countdown_placeholder.empty()
                
                # Rerun the app to check again
                if st.session_state.monitoring_active:
                    st.rerun()
    
    # Display the latest results with enhanced visualization
    with latest_results:
        # Add CSS for animations and styling
        st.markdown("""
        <style>
        @keyframes new-entry-pulse {
            0% { box-shadow: 0 0 0 0 rgba(255,0,0,0.7); }
            70% { box-shadow: 0 0 0 10px rgba(255,0,0,0); }
            100% { box-shadow: 0 0 0 0 rgba(255,0,0,0); }
        }
        .new-entry-pulse {
            animation: new-entry-pulse 1.5s ease-out 3;
            border-left: 5px solid #dc3545 !important;
        }
        .normal-entry {
            border-left: 5px solid #28a745;
        }
        .expander-anomaly {
            border-left: 5px solid #dc3545;
            background-color: #fff5f5;
        }
        .expander-normal {
            border-left: 5px solid #28a745;
            background-color: #f8fff8;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.subheader("Latest Detection Results")
        
        # Display a summary of results with more visual metrics
        if st.session_state.detection_results:
            # Add metrics at the top
            total_entries = len(st.session_state.detection_results)
            anomalous_entries = sum(1 for r in st.session_state.detection_results if r['result'][0])
            normal_entries = total_entries - anomalous_entries
            
            # Create a more visual dashboard
            col1, col2, col3 = st.columns(3)
            
            col1.markdown(f"""
            <div style="background-color: #e9ecef; padding: 12px; border-radius: 5px; text-align: center;">
                <div style="font-size: 0.8em; color: #666;">TOTAL ENTRIES</div>
                <div style="font-size: 1.8em; font-weight: bold;">{st.session_state.entry_count_total}</div>
                <div style="font-size: 0.9em;">Session: {total_entries}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate anomaly percentage
            anomaly_pct = 0 if total_entries == 0 else (anomalous_entries / total_entries) * 100
            
            col2.markdown(f"""
            <div style="background-color: #f8d7da; padding: 12px; border-radius: 5px; text-align: center;">
                <div style="font-size: 0.8em; color: #721c24;">ANOMALIES</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #dc3545;">{st.session_state.anomaly_count_total}</div>
                <div style="font-size: 0.9em;">Rate: {anomaly_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            col3.markdown(f"""
            <div style="background-color: #d4edda; padding: 12px; border-radius: 5px; text-align: center;">
                <div style="font-size: 0.8em; color: #155724;">NORMAL ACTIVITY</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #28a745;">{normal_entries}</div>
                <div style="font-size: 0.9em;">Rate: {100-anomaly_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add some spacing
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            
            # Create tabs for All, Anomalous, and Normal results
            tab1, tab2, tab3 = st.tabs(["All Results", "Anomalies Only", "Normal Activity"])
            
            with tab1:
                for idx, result_data in enumerate(reversed(st.session_state.detection_results)):
                    if idx >= 10:  # Show only the 10 most recent results
                        break
                    
                    # Create the main expander for each result
                    is_anomalous, score, reason = result_data['result']
                    expander_label = f"{result_data['timestamp']} - {result_data['detection_type']}"
                    
                    # Add visual indicator for anomalies in the expander label
                    if is_anomalous:
                        expander_label = f"⚠️ {expander_label} ⚠️"
                    
                    # Apply animation class for new entries
                    expander_class = result_data.get('animation_class', 'normal-entry')
                    if 'fresh' in result_data and result_data['fresh']:
                        # Remove the 'fresh' flag after first display
                        result_data['fresh'] = False
                    
                    # Apply different styling based on anomaly status
                    expander_style = "expander-anomaly" if is_anomalous else "expander-normal"
                    
                    with st.expander(expander_label, expanded=is_anomalous and 'fresh' in result_data and result_data.get('fresh', False)):
                        # Add custom CSS to the expander
                        st.markdown(f"""
                        <style>
                        .{expander_class} {{
                            border-left: 5px solid {('#dc3545' if is_anomalous else '#28a745')};
                        }}
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # Show result status and score
                        col1, col2 = st.columns(2)
                        with col1:
                            if is_anomalous:
                                st.markdown("""
                                <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px; text-align: center;">
                                    <div style="font-size: 1.2em; font-weight: bold; color: #721c24;">
                                        ⚠️ ANOMALY DETECTED ⚠️
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; text-align: center;">
                                    <div style="font-size: 1.2em; font-weight: bold; color: #155724;">
                                        ✅ NORMAL ACTIVITY
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            score_color = "red" if score > 0.7 else "orange" if score > 0.4 else "green"
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center;">
                                <div style="font-size: 0.8em; color: #666;">Anomaly Score</div>
                                <div style="font-size: 1.8em; font-weight: bold; color: {score_color};">{score:.2f}</div>
                                <div style="background-color: #e9ecef; height: 6px; border-radius: 3px; margin-top: 5px;">
                                    <div style="background-color: {score_color}; width: {score*100}%; height: 6px; border-radius: 3px;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display reason with styled box
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;">
                            <div style="font-size: 0.9em; color: #666;">DETECTION REASON:</div>
                            <div style="font-weight: 500;">{reason}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Use a checkbox instead of a nested expander
                        if st.checkbox(f"View Raw Data for {result_data['timestamp']}", key=f"raw_data_{idx}_all"):
                            st.json(result_data['raw_data'])
            
            with tab2:
                anomalous_results = [r for r in st.session_state.detection_results if r['result'][0]]
                if anomalous_results:
                    for idx, result_data in enumerate(reversed(anomalous_results)):
                        if idx >= 10:  # Show only the 10 most recent anomalies
                            break
                        
                        with st.expander(f"⚠️ {result_data['timestamp']} - {result_data['detection_type']} ⚠️", 
                                        expanded='fresh' in result_data and result_data.get('fresh', False)):
                            is_anomalous, score, reason = result_data['result']
                            
                            # Show result status and score
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("""
                                <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px; text-align: center;">
                                    <div style="font-size: 1.2em; font-weight: bold; color: #721c24;">
                                        ⚠️ ANOMALY DETECTED ⚠️
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                score_color = "red" if score > 0.7 else "orange" if score > 0.4 else "green"
                                st.markdown(f"""
                                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center;">
                                    <div style="font-size: 0.8em; color: #666;">Anomaly Score</div>
                                    <div style="font-size: 1.8em; font-weight: bold; color: {score_color};">{score:.2f}</div>
                                    <div style="background-color: #e9ecef; height: 6px; border-radius: 3px; margin-top: 5px;">
                                        <div style="background-color: {score_color}; width: {score*100}%; height: 6px; border-radius: 3px;"></div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Display reason with styled box
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;">
                                <div style="font-size: 0.9em; color: #666;">DETECTION REASON:</div>
                                <div style="font-weight: 500;">{reason}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Use a checkbox instead of a nested expander
                            if st.checkbox(f"View Raw Data for {result_data['timestamp']}", key=f"raw_data_{idx}_anomalous"):
                                st.json(result_data['raw_data'])
                else:
                    st.info("No anomalies detected yet")
            
            with tab3:
                normal_results = [r for r in st.session_state.detection_results if not r['result'][0]]
                if normal_results:
                    for idx, result_data in enumerate(reversed(normal_results)):
                        if idx >= 10:  # Show only the 10 most recent normal activities
                            break
                        
                        with st.expander(f"{result_data['timestamp']} - {result_data['detection_type']}"):
                            is_anomalous, score, reason = result_data['result']
                            
                            # Show result status and score
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("""
                                <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; text-align: center;">
                                    <div style="font-size: 1.2em; font-weight: bold; color: #155724;">
                                        ✅ NORMAL ACTIVITY
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                score_color = "red" if score > 0.7 else "orange" if score > 0.4 else "green"
                                st.markdown(f"""
                                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center;">
                                    <div style="font-size: 0.8em; color: #666;">Anomaly Score</div>
                                    <div style="font-size: 1.8em; font-weight: bold; color: {score_color};">{score:.2f}</div>
                                    <div style="background-color: #e9ecef; height: 6px; border-radius: 3px; margin-top: 5px;">
                                        <div style="background-color: {score_color}; width: {score*100}%; height: 6px; border-radius: 3px;"></</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display reason with styled box
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;">
                                <div style="font-size: 0.9em; color: #666;">DETECTION REASON:</div>
                                <div style="font-weight: 500;">{reason}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Use a checkbox instead of a nested expander
                            if st.checkbox(f"View Raw Data for {result_data['timestamp']}", key=f"raw_data_{idx}_normal"):
                                st.json(result_data['raw_data'])
                else:
                    st.info("No normal activities recorded yet")
        
        else:
            st.info("No detection results yet. Start monitoring to see results here.")
            
            # Add a demo entry when no results exist
            if st.button("Show Demo Result"):
                # Create a fake demo result
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Determine if demo should be anomalous (random)
                is_anomalous = random.choice([True, False])
                
                # Create appropriate demo data
                if monitor_collection == "purchases":
                    detection_type = "Purchase Anomaly"
                    if is_anomalous:
                        score = random.uniform(0.75, 0.95)
                        reason = "Unusually large purchase amount that deviates significantly from user's history."
                        raw_data = {
                            "user_id": "user123",
                            "amount": 999.99,
                            "currency": "USD",
                            "payment_method": "credit_card",
                            "purchase_time": timestamp,
                            "product_category": "electronics",
                            "doc_id": "demo_purchase_001"
                        }
                    else:
                        score = random.uniform(0.05, 0.25)
                        reason = "Purchase amount is within normal range for user."
                        raw_data = {
                            "user_id": "user123",
                            "amount": 24.99,
                            "currency": "USD",
                            "payment_method": "credit_card",
                            "purchase_time": timestamp,
                            "product_category": "books",
                            "doc_id": "demo_purchase_001"
                        }
                else:  # logins
                    detection_type = "Login Anomaly"
                    if is_anomalous:
                        score = random.uniform(0.75, 0.95)
                        reason = "Login from unusual location and device not previously associated with user."
                        raw_data = {
                            "user_id": "user123",
                            "ip_address": "203.0.113.42",
                            "device_id": "unknown_device_789",
                            "login_time": timestamp,
                            "success": True,
                            "location": "Unknown Location",
                            "doc_id": "demo_login_001"
                        }
                    else:
                        score = random.uniform(0.05, 0.25)
                        reason = "Login from recognized device and location."
                        raw_data = {
                            "user_id": "user123",
                            "ip_address": "192.168.1.1",
                            "device_id": "iphone_123",
                            "login_time": timestamp,
                            "success": True,
                            "location": "San Francisco, CA",
                            "doc_id": "demo_login_001"
                        }
                
                # Store the demo result
                animation_class = "new-entry-pulse" if is_anomalous else "normal-entry"
                st.session_state.detection_results.append({
                    "timestamp": timestamp,
                    "detection_type": detection_type,
                    "result": (is_anomalous, score, reason),
                    "raw_data": raw_data,
                    "animation_class": animation_class,
                    "fresh": True,
                    "demo": True
                })
                
                # Update counters
                st.session_state.entry_count_total += 1
                if is_anomalous:
                    st.session_state.anomaly_count_total += 1
                
                # Show alert message for demo
                if is_anomalous:
                    st.session_state.alert_message = f"⚠️ DEMO: Anomaly detected!"
                    st.session_state.alert_level = "error"
                else:
                    st.session_state.alert_message = f"👁️ DEMO: Normal activity detected"
                    st.session_state.alert_level = "info"
                
                st.session_state.alert_shown = True
                
                # Rerun to show the demo result
                st.rerun()

    # Add a clear results button at the bottom
    if st.session_state.detection_results:
        if st.button("Clear All Results"):
            st.session_state.detection_results = []
            st.session_state.entry_count_total = 0
            st.session_state.anomaly_count_total = 0
            st.session_state.alert_shown = False
            st.session_state.alert_message = ""
            st.rerun() 

# Function to handle new document changes from Firestore
def on_snapshot(changes, collection_name, detector):
    for change in changes:
        if change.type.name == 'ADDED':  # Only process new documents
            try:
                doc_data = change.document.to_dict()
                doc_data['doc_id'] = change.document.id
                
                # Process entry based on collection type
                if collection_name == "purchases":
                    result = detector.detect_unusual_purchase(doc_data)
                    detection_type = "Purchase Anomaly"
                else:  # logins
                    result = detector.detect_unusual_login(doc_data)
                    detection_type = "Login Anomaly"
                
                # Store the result with timestamp and raw data
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.detection_results.append({
                    "timestamp": timestamp,
                    "detection_type": detection_type,
                    "result": result,
                    "raw_data": doc_data
                })
                
                # Save to anomaly_results collection if anomalous
                is_anomalous, score, reason = result
                if is_anomalous:
                    try:
                        db = initialize_firebase()
                        result_data = {
                            "detection_type": "purchase" if collection_name == "purchases" else "login",
                            "user_id": doc_data.get("user_id", "unknown"),
                            "is_anomalous": is_anomalous,
                            "anomaly_score": score,
                            "reason": reason,
                            "timestamp": timestamp,
                            f"{collection_name[:-1]}_data": doc_data
                        }
                        db.collection("anomaly_results").add(result_data)
                    except Exception as e:
                        print(f"Error saving anomaly result: {e}")
                
                # Signal that new data is available
                st.session_state.new_data_available = True
            except Exception as e:
                print(f"Error processing document: {e}")
    
    # Signal that callback is done
    st.session_state.callback_done.set()

# Function to start the Firestore collection listener
def start_collection_listener(collection_name, detector):
    # Stop existing listener if there is one
    if 'listener_unsubscribe' in st.session_state:
        st.session_state.listener_unsubscribe()
    
    try:
        db = initialize_firebase()
        
        # Get the last timestamp to use as query constraint
        query = db.collection(collection_name)
        
        if st.session_state.last_timestamp:
            # Query for entries after the last timestamp
            query = query.where('timestamp', '>', st.session_state.last_timestamp)
        
        # Order by timestamp for real-time updates
        query = query.order_by('timestamp')
        
        # Create and set the snapshot listener
        st.session_state.callback_done = threading.Event()
        st.session_state.listener_unsubscribe = query.on_snapshot(
            lambda doc_snapshot, changes, read_time: 
            on_snapshot(doc_snapshot, changes, read_time, collection_name, detector)
        )
        
    except Exception as e:
        st.error(f"Error setting up collection listener: {e}")

# Function to fetch data from Firebase
def fetch_firebase_data(collection_name, limit=1000):
    """
    Fetch data from a Firestore collection
    
    Parameters:
    collection_name (str): Name of the collection to fetch from
    limit (int): Maximum number of documents to retrieve
    
    Returns:
    pd.DataFrame: DataFrame containing the data
    """
    db = initialize_firebase()
    docs = db.collection(collection_name).limit(limit).get()
    
    data = []
    for doc in docs:
        doc_data = doc.to_dict()
        doc_data['doc_id'] = doc.id  # Include the document ID
        data.append(doc_data)
    
    return pd.DataFrame(data) if data else pd.DataFrame()

# Function to generate preset data (if Firebase data is not available)
def generate_preset_data():
    # Sample purchase data
    purchase_data = pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
        'amount': [100, 150, 120, 500, 550, 50, 60, 55, 200],
        'category': ['Electronics', 'Electronics', 'Books', 'Furniture', 'Furniture', 'Food', 'Food', 'Food', 'Electronics'],
        'timestamp': pd.date_range(start='2023-01-01', periods=9, freq='D')
    })
    
    # Sample login data
    login_data = pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3],
        'latitude': [40.7128, 40.7130, 40.7135, 34.0522, 34.0530, 51.5074, 51.5080, 48.8566],
        'longitude': [-74.0060, -74.0065, -74.0070, -118.2437, -118.2440, -0.1278, -0.1275, 2.3522],
        'timestamp': pd.date_range(start='2023-01-01', periods=8, freq='D')
    })
    
    return purchase_data, login_data

# Train model or load if available
@st.cache_resource
def get_trained_model():
    model_path = 'crm_anomaly_detector.pkl'
    
    if os.path.exists(model_path):
        try:
            # Load existing model
            with st.spinner('Loading pre-trained model...'):
                detector = CRMAnomalyDetector.load_model(model_path)
                return detector
        except Exception as e:
            st.warning(f"Failed to load existing model: {e}. Training new model...")
    
    # Try to get data from Firebase
    with st.spinner('Retrieving data from Firebase...'):
        try:
            purchase_data = fetch_firebase_data('purchases')
            login_data = fetch_firebase_data('logins')
            
            # Check if we got data
            if purchase_data.empty or login_data.empty:
                st.warning("No data found in Firebase, using preset data instead")
                purchase_data, login_data = generate_preset_data()
            else:
                st.success(f"Successfully retrieved {len(purchase_data)} purchases and {len(login_data)} logins from Firebase")
                
                # Ensure data has the right columns
                required_purchase_cols = ['user_id', 'amount', 'category', 'timestamp']
                required_login_cols = ['user_id', 'latitude', 'longitude', 'timestamp']
                
                # Convert data types if needed
                if 'user_id' in purchase_data and purchase_data['user_id'].dtype == 'object':
                    purchase_data['user_id'] = purchase_data['user_id'].astype(int)
                if 'user_id' in login_data and login_data['user_id'].dtype == 'object':
                    login_data['user_id'] = login_data['user_id'].astype(int)
                
                # Check if we have the required columns
                missing_purchase = [col for col in required_purchase_cols if col not in purchase_data.columns]
                missing_login = [col for col in required_login_cols if col not in login_data.columns]
                
                if missing_purchase or missing_login:
                    st.warning(f"Firebase data missing required columns. Using preset data instead.")
                    purchase_data, login_data = generate_preset_data()
                
        except Exception as e:
            st.error(f"Error retrieving Firebase data: {e}")
            st.warning("Using preset data instead")
            purchase_data, login_data = generate_preset_data()
    
    # Train the model with the data
    with st.spinner('Training model on the data...'):
        detector = CRMAnomalyDetector()
        detector.fit_purchase_model(purchase_data)
        detector.fit_location_model(login_data)
        detector.save_model(model_path)
        
    return detector

# UI helper functions
def display_result(result, detection_type):
    is_anomalous, score, reason = result
    
    col1, col2 = st.columns(2)
    
    with col1:
        if is_anomalous:
            st.error("⚠️ ANOMALY DETECTED")
        else:
            st.success("✅ NORMAL ACTIVITY")
    
    with col2:
        # Display score gauge
        score_color = "red" if score > 0.7 else "orange" if score > 0.4 else "green"
        st.markdown(f"""
        <div style="text-align: center;">
            <p style="margin-bottom: 0;">Anomaly Score</p>
            <h2 style="color: {score_color}; margin: 0;">{score:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"**Reason:** {reason}")
    
    # Show detailed info
    with st.expander("View Full Analysis Details"):
        st.write("Detection Type:", detection_type)
        st.write("Anomalous:", is_anomalous)
        st.write("Anomaly Score:", score)
        st.write("Explanation:", reason)

def save_results_to_firebase(data, collection_name):
    """Save detection results to Firebase"""
    try:
        db = initialize_firebase()
        collection_ref = db.collection(collection_name)
        
        # Add timestamp for when the result was saved
        data['saved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Add to Firebase
        collection_ref.add(data)
        
        return True, "Data saved successfully!"
    except Exception as e:
        return False, f"Error saving to Firebase: {str(e)}"

def load_detector_and_run_app():
    # Sidebar for Firebase options
    st.sidebar.title("🔄 Firebase Options")
    
    # Initialize Firebase
    try:
        db = initialize_firebase()
        st.sidebar.success("✅ Connected to Firebase")
        
        # Option to view raw data
        if st.sidebar.checkbox("View Raw Firebase Data"):
            st.sidebar.subheader("Select Collection")
            collection_choice = st.sidebar.selectbox(
                "Choose a collection to view",
                ["purchases", "logins", "anomaly_results"]
            )
            
            records_limit = st.sidebar.slider("Number of records to fetch", 10, 1000, 100)
            
            if st.sidebar.button("Fetch Data"):
                with st.spinner(f"Fetching data from {collection_choice}..."):
                    data = fetch_firebase_data(collection_choice, records_limit)
                    if not data.empty:
                        st.subheader(f"Raw data from '{collection_choice}' collection")
                        st.dataframe(data)
                    else:
                        st.warning(f"No data found in the '{collection_choice}' collection")
    except Exception as e:
        st.sidebar.error(f"Firebase initialization error: {str(e)}")
    st.sidebar.title("🔍 Detection Modes")
    detection_mode = st.sidebar.radio(
        "Select Detection Mode",
        ["Manual Testing", "Real-Time Monitoring"]
    )
    # Load and train the model
    detector = get_trained_model()
    
    # Create Streamlit app
    st.title("🔍 CRM Anomaly Detector")
    st.markdown("""
    This application detects unusual patterns in CRM data using real data from Firebase including:
    - Suspicious purchase patterns
    - Unusual login locations
    
    Select a detection type below and enter the required information.
    """)
    if detection_mode == "Manual Testing":
    # Create tabs for different detection types
        tab1, tab2 = st.tabs(["Purchase Anomaly Detection", "Login Location Anomaly Detection"])
        
        # Purchase anomaly detection
        with tab1:
            st.header("Detect Unusual Purchase Patterns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                user_id = st.number_input("User ID", min_value=1, step=1, value=1)
                amount = st.number_input("Purchase Amount ($)", min_value=0.0, step=10.0, value=100.0)
            
            with col2:
                category = st.selectbox(
                    "Product Category", 
                    options=["Electronics", "Books", "Furniture", "Food", "Clothing", "Jewelry", "Travel", "Entertainment", "Other"]
                )
                purchase_date = st.date_input("Purchase Date", value=datetime.now().date())
                purchase_time = st.time_input("Purchase Time", value=datetime.now().time())
            
            timestamp = datetime.combine(purchase_date, purchase_time)
            
            # JSON input option
            st.markdown("### Or Input JSON Data")
            json_input = st.text_area(
                "Enter purchase data in JSON format",
                value=json.dumps({
                    "user_id": user_id,
                    "amount": amount,
                    "category": category,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
                }, indent=2),
                height=150
            )
            
            # Option to get data from Firebase
            use_firebase_data = st.checkbox("Fetch latest purchase from Firebase")
            
            if use_firebase_data:
                firebase_user_id = st.number_input("Firebase User ID", min_value=1, step=1, value=user_id)
                if st.button("Fetch Latest Purchase"):
                    try:
                        db = initialize_firebase()
                        # Get the most recent purchase for this user
                        query = db.collection('purchases').where('user_id', '==', firebase_user_id).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1)
                        docs = query.get()
                        
                        if docs:
                            purchase_data = docs[0].to_dict()
                            # Update the JSON input
                            json_input = json.dumps(purchase_data, indent=2)
                            st.success(f"Latest purchase data fetched for user {firebase_user_id}")
                        else:
                            st.warning(f"No purchase data found for user {firebase_user_id}")
                    except Exception as e:
                        st.error(f"Error fetching data: {str(e)}")
            
            if st.button("Detect Purchase Anomaly"):
                try:
                    # Try to use JSON input if possible
                    try:
                        purchase_data = json.loads(json_input)
                    except:
                        # Fall back to form input
                        purchase_data = {
                            "user_id": user_id,
                            "amount": amount,
                            "category": category,
                            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    
                    # Run detection
                    result = detector.detect_unusual_purchase(purchase_data)
                    
                    # Display result
                    st.markdown("## Detection Result")
                    display_result(result, "Purchase Anomaly")
                    
                    # Save results to Firebase if connected
                    if 'db' in locals():
                        save_to_firebase = st.checkbox("Save this result to Firebase", value=True)
                        if save_to_firebase:
                            result_data = {
                                "detection_type": "purchase",
                                "user_id": purchase_data["user_id"],
                                "is_anomalous": result[0],
                                "anomaly_score": result[1],
                                "reason": result[2],
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "purchase_data": purchase_data
                            }
                            
                            success, message = save_results_to_firebase(result_data, "anomaly_results")
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                    
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
                    st.info("Please check your input format and try again.")
        
        # Login location anomaly detection
        with tab2:
            st.header("Detect Suspicious Login Locations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                login_user_id = st.number_input("User ID", min_value=1, step=1, value=1, key="login_user_id")
                latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=40.7128)
                longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-74.0060)
            
            with col2:
                login_date = st.date_input("Login Date", value=datetime.now().date(), key="login_date")
                login_time = st.time_input("Login Time", value=datetime.now().time(), key="login_time")
                
                # Optional previous login info
                has_prev_login = st.checkbox("Include previous login information")
            
            login_timestamp = datetime.combine(login_date, login_time)
            
            # Previous login details
            if has_prev_login:
                st.markdown("### Previous Login Details")
                col1, col2 = st.columns(2)
                
                with col1:
                    prev_latitude = st.number_input("Previous Latitude", min_value=-90.0, max_value=90.0, value=40.7128)
                    prev_longitude = st.number_input("Previous Longitude", min_value=-180.0, max_value=180.0, value=-74.0060)
                    
                with col2:
                    prev_login_date = st.date_input("Previous Login Date", value=datetime.now().date() - pd.Timedelta(days=1))
                    prev_login_time = st.time_input("Previous Login Time", value=datetime.now().time())
                
                prev_timestamp = datetime.combine(prev_login_date, prev_login_time)
            
            # JSON input option
            st.markdown("### Or Input JSON Data")
            login_json = {}
            if has_prev_login:
                login_json = {
                    "user_id": login_user_id,
                    "latitude": latitude,
                    "longitude": longitude,
                    "timestamp": login_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "last_login_timestamp": prev_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "last_login_location": (prev_latitude, prev_longitude)
                }
            else:
                login_json = {
                    "user_id": login_user_id,
                    "latitude": latitude,
                    "longitude": longitude,
                    "timestamp": login_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                }
                
            json_login_input = st.text_area(
                "Enter login data in JSON format",
                value=json.dumps(login_json, indent=2),
                height=150
            )

            use_firebase_login = st.checkbox("Fetch latest login from Firebase")
            if use_firebase_login:
                firebase_login_user_id = st.number_input("Firebase User ID for Login", min_value=1, step=1, value=login_user_id)
                if st.button("Fetch Latest Login"):
                    try:
                        db = initialize_firebase()
                        # Get the most recent login for this user
                        query = db.collection('logins').where('user_id', '==', firebase_login_user_id).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1)
                        docs = query.get()
                        
                        if docs:
                            login_data = docs[0].to_dict()
                            
                            # If we want previous login info, get the second most recent login
                            if has_prev_login:
                                query_prev = db.collection('logins').where('user_id', '==', firebase_login_user_id).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(2)
                                docs_prev = query_prev.get()
                                
                                if len(docs_prev) > 1:
                                    prev_login = docs_prev[1].to_dict()
                                    login_data['last_login_timestamp'] = prev_login['timestamp']
                                    login_data['last_login_location'] = (prev_login['latitude'], prev_login['longitude'])
                            
                            # Update the JSON input
                            json_login_input = json.dumps(login_data, indent=2)
                            st.success(f"Latest login data fetched for user {firebase_login_user_id}")
                        else:
                            st.warning(f"No login data found for user {firebase_login_user_id}")
                    except Exception as e:
                        st.error(f"Error fetching data: {str(e)}")
            
            if st.button("Detect Login Anomaly"):
                try:
                    # Try to use JSON input if possible
                    try:
                        login_data = json.loads(json_login_input)
                    except:
                        # Fall back to form input
                        login_data = {
                            "user_id": login_user_id,
                            "latitude": latitude,
                            "longitude": longitude,
                            "timestamp": login_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        if has_prev_login:
                            login_data["last_login_timestamp"] = prev_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                            login_data["last_login_location"] = (prev_latitude, prev_longitude)
                    
                    # Run detection
                    result = detector.detect_unusual_login(login_data)
                    
                    # Display result
                    st.markdown("## Detection Result")
                    display_result(result, "Login Location Anomaly")
                    
                    # Save results to Firebase if connected
                    if 'db' in locals():
                        save_to_firebase = st.checkbox("Save this result to Firebase", value=True, key="save_login_result")
                        if save_to_firebase:
                            result_data = {
                                "detection_type": "login",
                                "user_id": login_data["user_id"],
                                "is_anomalous": result[0],
                                "anomaly_score": result[1],
                                "reason": result[2],
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "login_data": login_data
                            }
                            
                            success, message = save_results_to_firebase(result_data, "anomaly_results")
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                    
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
                    st.info("Please check your input format and try again.")
    else:
        setup_real_time_monitoring(detector)   
    # Add a section for model information
    st.sidebar.title("🧠 Model Information")
    with st.sidebar.expander("About the Model"):
        st.markdown("""
        **CRM Anomaly Detector** uses machine learning to identify unusual patterns in customer data:
        
        - **Purchase Anomaly Detection:** Identifies unusual spending patterns based on amount, category, and timing
        - **Login Location Detection:** Identifies logins from unusual locations or impossible travel patterns
        
        The model is trained using historical CRM data and uses Isolation Forest algorithm and distance-based detection.
        """)
    
    # Add a visualization section
    if st.sidebar.checkbox("Enable Visualization"):
        st.header("📊 Data Visualization")
        
        viz_tab1, viz_tab2 = st.tabs(["Purchase Patterns", "Login Locations"])
        
        with viz_tab1:
            st.subheader("Purchase Pattern Visualization")
            st.markdown("This section would display charts of purchase patterns and anomalies.")
            
            # Placeholder for future visualization code
            st.info("Visualization features will be implemented in a future update.")
            
            # Example visualization placeholder
            chart_data = pd.DataFrame(
                np.random.randn(20, 3),
                columns=['Normal Purchases', 'Suspicious Purchases', 'Threshold']
            )
            st.line_chart(chart_data)
        
        with viz_tab2:
            st.subheader("Login Location Map")
            st.markdown("This section would display a map of login locations with anomalies highlighted.")
            
            # Placeholder for future map visualization
            st.info("Map visualization will be implemented in a future update.")
            
            # Example map data visualization placeholder
            map_data = pd.DataFrame(
                np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
                columns=['lat', 'lon']
            )
            st.map(map_data)
    
    # Add a feedback section
    st.sidebar.title("📝 Feedback")
    if st.sidebar.button("Submit Feedback"):
        st.sidebar.success("Thank you for your feedback!")
    
    # Add app information in the footer
    st.markdown("---")
    st.markdown("CRM Anomaly Detector v1.0 | © 2023")

# Run the app
if __name__ == "__main__":
    load_detector_and_run_app()