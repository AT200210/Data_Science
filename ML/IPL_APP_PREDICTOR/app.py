import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(
    page_title="IPL Match Predictor",
    page_icon="🏏",
    layout="wide"
)

# Title and description
st.title("IPL Match Outcome Predictor 🏏")
st.markdown("""
This app predicts the probability of a team winning an IPL match based on current match conditions.
Enter the details below to get a prediction!
""")

# Function to load models and metadata
@st.cache_resource
def load_models():
    try:
        # Attempt to load team and city data from file
        with open('teams_cities.pkl', 'rb') as f:
            data = pickle.load(f)
            teams = data.get('teams', [])
            cities = data.get('cities', [])
            model_results = data.get('model_results', {
                'logistic_regression': {'filename': 'ipl_predictor_logistic_regression.pkl'},
                'decision_tree': {'filename': 'ipl_predictor_decision_tree.pkl'},
                'random_forest': {'filename': 'ipl_predictor_random_forest.pkl'}
            })
    except FileNotFoundError:
        st.error("""
        teams_cities.pkl not found! Please run the setup or training script to generate required files.
        
        If you're testing this app without actual data, consider using a demo mode.
        """)
        return None, None, None, None

    # Load all models from the given filenames
    models = {}
    missing_files = False

    for model_name, info in model_results.items():
        model_file = info['filename']
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                models[model_name] = pickle.load(f)
        else:
            missing_files = True

    # Try fallback to default logistic regression model if none loaded
    if len(models) == 0 and os.path.exists('ipl_predictor_model.pkl'):
        with open('ipl_predictor_model.pkl', 'rb') as f:
            models['logistic_regression'] = pickle.load(f)

    if len(models) == 0 or missing_files:
        return None, None, None, None

    return models, teams, cities, model_results
# Function to create demo models
@st.cache_resource
def create_demo_models():
    teams = [
        'Royal Challengers Bengaluru',
        'Mumbai Indians',
        'Kolkata Knight Riders',
        'Rajasthan Royals',
        'Chennai Super Kings',
        'Sunrisers Hyderabad',
        'Delhi Capitals', 
        'Punjab Kings',
        'Lucknow Super Giants',
        'Gujarat Titans'
    ]
    
    cities = ['Mumbai', 'Kolkata', 'Chennai', 'Bengaluru', 'Delhi', 'Hyderabad', 'Ahmedabad', 'Pune']
    
    # Create a placeholder dataframe for fitting the transformer
    

    dummy_X = pd.DataFrame({
    'batting_team': ['Mumbai Indians', 'Chennai Super Kings'],
    'bowling_team': ['Chennai Super Kings', 'Mumbai Indians'],
    'city': ['Mumbai', 'Chennai'],
    'runs_left': [50, 30],
    'balls_left': [30, 24],
    'wickets_left': [5, 3],
    'total_runs_x': [180, 160],
    'crr': [8.0, 7.5],
    'rrr': [10.0, 7.5]
    })

    dummy_y = pd.Series([1, 0])
    
    
    
    trf = ColumnTransformer([
        ('categorical', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
    ], remainder='passthrough')
    
    models = {}
    
    # Logistic Regression
    pipe_lr = Pipeline([
        ('transform', trf),
        ('model', LogisticRegression(solver='liblinear'))
    ])
    pipe_lr.fit(dummy_X, dummy_y)
    models['logistic_regression'] = pipe_lr
    
    # Decision Tree
    pipe_dt = Pipeline([
        ('transform', trf),
        ('model', DecisionTreeClassifier(max_depth=3))
    ])
    pipe_dt.fit(dummy_X, dummy_y)
    models['decision_tree'] = pipe_dt
    
    # Random Forest
    pipe_rf = Pipeline([
        ('transform', trf),
        ('model', RandomForestClassifier(n_estimators=10, max_depth=3))
    ])
    pipe_rf.fit(dummy_X, dummy_y)
    models['random_forest'] = pipe_rf
    
    model_results = {
        'logistic_regression': {'accuracy': 0.75, 'filename': 'demo_lr.pkl'},
        'decision_tree': {'accuracy': 0.70, 'filename': 'demo_dt.pkl'},
        'random_forest': {'accuracy': 0.80, 'filename': 'demo_rf.pkl'}
    }
    
    return models, teams, cities, model_results

# Try to load the models
models, teams, cities, model_results = load_models()

# If models not found, provide option to use demo models
if models is None:
    if st.button('Use Demo Models'):
        models, teams, cities, model_results = create_demo_models()
        st.success("Demo models created! These are for demonstration purposes only and predictions won't be accurate.")

# Only continue if we have models
if models is not None:
    # Model selection
    model_options = list(models.keys())
    
    # Format model names for display
    display_names = {
        'logistic_regression': 'Logistic Regression',
        'decision_tree': 'Decision Tree',
        'random_forest': 'Random Forest'
    }
    
    # Add accuracy info if available
    model_display_options = []
    for model_name in model_options:
        if model_name in model_results and 'accuracy' in model_results[model_name]:
            acc = model_results[model_name]['accuracy']
            display_name = f"{display_names.get(model_name, model_name)} (Accuracy: {acc:.2f})"
        else:
            display_name = display_names.get(model_name, model_name)
        model_display_options.append((model_name, display_name))
    
    # Convert to format for selectbox
    model_keys = [m[0] for m in model_display_options]
    model_display = [m[1] for m in model_display_options]
    
    # Model selection dropdown
    selected_model_display = st.selectbox('Select Prediction Model', model_display)
    selected_model_index = model_display.index(selected_model_display)
    selected_model_name = model_keys[selected_model_index]
    selected_model = models[selected_model_name]
    
    # Create two columns for team selection
    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox('Batting Team', sorted(teams))
        
    with col2:
        bowling_team = st.selectbox('Bowling Team', sorted(teams))

    # Display warning if same team is selected
    if batting_team == bowling_team:
        st.warning("Please select different teams for batting and bowling")

    # Match conditions
    st.subheader("Match Conditions")

    col1, col2, col3 = st.columns(3)

    with col1:
        city = st.selectbox('City', sorted(cities))
        
    with col2:
        target = st.number_input('Target Score', min_value=1, max_value=250, value=180)

    with col3:
        total_overs = st.number_input('Total Overs', min_value=1, max_value=20, value=20)
        total_balls = total_overs * 6

    # Current situation
    st.subheader("Current Match Situation")

    col1, col2, col3 = st.columns(3)

    with col1:
        current_score = st.number_input('Current Score', min_value=0, max_value=250, value=100)
        runs_left = target - current_score

    with col2:
        overs_completed = st.slider('Overs Completed', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
        balls_completed = int(overs_completed * 6)
        balls_left = total_balls - balls_completed
        
    with col3:
        wickets = st.slider('Wickets Lost', min_value=0, max_value=10, value=2)
        wickets_left = 10 - wickets

    # Calculate CRR and RRR
    if balls_completed > 0:
        crr = (current_score * 6) / balls_completed
    else:
        crr = 0

    if balls_left > 0:
        rrr = (runs_left * 6) / balls_left
    else:
        rrr = 0

    # Display metrics
    st.subheader("Match Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Runs Left", runs_left)
        
    with col2:
        st.metric("Balls Left", balls_left)
        
    with col3:
        st.metric("Current Run Rate", round(crr, 2))
        
    with col4:
        st.metric("Required Run Rate", round(rrr, 2))

    # Make prediction
    if st.button('Predict Outcome'):
        if batting_team == bowling_team:
            st.error("Batting and bowling teams cannot be the same!")
        elif balls_left <= 0:
            st.error("No balls left to play! The match is already over.")
        elif runs_left <= 0:
            st.success(f"{batting_team} has already won the match!")
        else:
            # Create a DataFrame with the input values
            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets_left': [wickets_left],
                'total_runs_x': [target],
                'crr': [crr],
                'rrr': [rrr]
            })
            
            # Make prediction with selected model
            result = selected_model.predict_proba(input_df)
            
            # Display prediction
            st.subheader(f"Match Prediction ({display_names.get(selected_model_name, selected_model_name)})")
            
            batting_win_probability = result[0][1] * 100
            bowling_win_probability = result[0][0] * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(f"{batting_team} Win Probability", f"{batting_win_probability:.2f}%")
                
            with col2:
                st.metric(f"{bowling_team} Win Probability", f"{bowling_win_probability:.2f}%")
            
            # Visual representation
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh(['Win Probability'], [batting_win_probability], color='green', label=batting_team)
            ax.barh(['Win Probability'], [bowling_win_probability], color='red', left=[batting_win_probability], label=bowling_team)
            ax.set_xlim(0, 100)
            ax.legend()
            ax.set_xlabel('Probability (%)')
            ax.set_yticks([])
            
            st.pyplot(fig)
            
            # Provide analysis
            st.subheader("Analysis")
            
            if batting_win_probability > bowling_win_probability:
                st.write(f"**{batting_team}** has a better chance of winning the match.")
            else:
                st.write(f"**{bowling_team}** has a better chance of winning the match.")
            
            if rrr > 12:
                st.write("The required run rate is very high, which puts significant pressure on the batting team.")
            elif rrr > 8:
                st.write("The required run rate is challenging but achievable with good batting.")
            else:
                st.write("The required run rate is manageable for the batting team.")
                
            if wickets_left <= 3:
                st.write("With few wickets remaining, the batting team needs to be cautious.")
            
            if balls_left < 36:
                st.write("In the final overs, the match can swing either way with a few big hits.")

# Add footer with information
st.markdown("---")
st.markdown("### About the App")
st.write("""
This IPL Match Prediction App uses machine learning to predict match outcomes based on current match situations.
Multiple prediction models are available, each with different characteristics:

- **Logistic Regression**: Good at capturing linear relationships between features
- **Decision Tree**: Captures non-linear patterns and is easy to interpret
- **Random Forest**: An ensemble method that often provides more robust predictions
""")

# Additional information about using the app
with st.expander("How to use this app"):
    st.write("""
    1. Select the machine learning model you want to use for prediction
    2. Select the batting and bowling teams
    3. Choose the city where the match is being played
    4. Enter the target score and total overs
    5. Input the current match situation (score, overs completed, wickets lost)
    6. Click on 'Predict Outcome' to see the prediction
    
    The app will calculate the win probability for both teams based on the current match situation.
    
    Try comparing predictions from different models to gain better insights!
    """)

# Instructions for first-time setup
with st.expander("First-time setup"):
    st.write("""
    Before running this app for the first time:
    
    1. Make sure you have the required data files (`matches.csv` and `deliveries.csv`) in the same directory
    2. Run `train_model.py` to train and save the models
    3. Once the models are saved, restart this Streamlit app
    
    If you don't have the data files, you can still use the demo models for demonstration purposes.
    """)