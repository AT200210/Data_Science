import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("Loading and preprocessing data...")

# Load datasets
try:
    # Load your datasets (replace with actual file paths)
    matches = pd.read_csv("Datasets/matches.csv")
    delivery = pd.read_csv("Datasets/deliveries.csv")
    
    # Data preprocessing
    total_score_df = delivery.groupby(["match_id", "inning"]).sum()['total_runs'].reset_index()
    total_score_df = total_score_df[total_score_df['inning'] == 1]
    
    match_df = matches.merge(total_score_df[['match_id','total_runs']], left_on='id', right_on='match_id')
    
    # Team name standardization
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
    
    match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
    match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
    match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    match_df['team1'] = match_df['team1'].str.replace('Kings XI Punjab', 'Punjab Kings')
    match_df['team2'] = match_df['team2'].str.replace('Kings XI Punjab', 'Punjab Kings')
    match_df['team1'] = match_df['team1'].str.replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')
    match_df['team2'] = match_df['team2'].str.replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')
    
    match_df = match_df[match_df['team1'].isin(teams)]
    match_df = match_df[match_df['team2'].isin(teams)]
    match_df = match_df[match_df['method'] != 'D/L']
    match_df = match_df[['match_id', 'city', 'winner', 'total_runs']]
    
    delivery_df = match_df.merge(delivery, on='match_id')
    delivery_df = delivery_df[delivery_df['inning'] == 2]
    
    delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()
    delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']
    delivery_df['balls_left'] = 126 - (delivery_df['over']*6 + delivery_df['ball'])
    
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: x if x == '0' else "1")
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')
    wickets = delivery_df.groupby("match_id")['player_dismissed'].cumsum().values
    delivery_df['wickets_left'] = 10 - wickets
    
    delivery_df['crr'] = (delivery_df['current_score']*6)/(120 - delivery_df['balls_left'])
    delivery_df['rrr'] = (delivery_df['runs_left']*6)/delivery_df['balls_left']
    
    delivery_df['result'] = delivery_df.apply(lambda row: 1 if row['batting_team'] == row['winner'] else 0, axis=1)
    
    final_df = delivery_df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 
                           'wickets_left', 'total_runs_x', 'crr', 'rrr', 'result']]
    final_df = final_df.sample(final_df.shape[0])
    
    final_df['crr'].replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(subset=['crr'], inplace=True)
    final_df = final_df[final_df['balls_left'] != 0]
    final_df.dropna(inplace=True)
    
    X = final_df.iloc[:, :-1]
    y = final_df['result']
    
    print("Splitting data and training models...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=40)
    
    # Create a column transformer for preprocessing
    trf = ColumnTransformer([
        ('categorical', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
    ], remainder='passthrough')
    
    # Define all models to train
    models = {
        'logistic_regression': LogisticRegression(solver='liblinear', max_iter=1000),
        'decision_tree': DecisionTreeClassifier(max_depth=20, random_state=42, criterion='entropy'),
        'random_forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, criterion='entropy')
    }
    
    # Train and save each model
    results = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        # Create pipeline with preprocessing and model
        pipe = Pipeline([
            ('transform', trf),
            ('model', model)
        ])
        
        # Train the model
        pipe.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = pipe.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{model_name} accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Save model
        model_filename = f"ipl_predictor_{model_name}.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(pipe, f)
        print(f"Model saved as {model_filename}")
        
        # Store results
        results[model_name] = {
            'accuracy': accuracy,
            'filename': model_filename
        }
    
    # Save teams and cities for later use
    teams_cities = {
        'teams': teams,
        'cities': final_df['city'].unique().tolist(),
        'model_results': results
    }
    
    with open('teams_cities.pkl', 'wb') as f:
        pickle.dump(teams_cities, f)
    
    print("\nTraining complete! All models saved.")
    print("\nModel comparison:")
    for model_name, res in results.items():
        print(f"{model_name}: {res['accuracy']:.4f}")
    
    # Identify best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest model: {best_model[0]} with accuracy {best_model[1]['accuracy']:.4f}")
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure that matches.csv and deliveries.csv are in the current directory.")