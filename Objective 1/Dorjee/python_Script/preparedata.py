"""
preparedata.py
--------------
Data Wrangling, Preparation, and Sampling Pipeline

Analytical Question:
"Did matches in the Knockout Stage have significantly more fouls on average than matches in the Group Stage?"
"""

import os
import pandas as pd

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'Rawdata.csv')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

os.makedirs(PROCESSED_DIR, exist_ok=True)


def wrangle_data(raw_csv_path):
    """
    Reads raw match dataset and converts it to tidy team-match observations.
    Raw data is accessed strictly read-only.
    """
    print("\n" + "="*70)
    print("1. DATA WRANGLING")
    print("="*70)
    
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_csv_path}")
    
    raw_df = pd.read_csv(raw_csv_path)
    print(f"Loaded raw dataset from: {raw_csv_path}")
    print(f"Raw matches: {len(raw_df)}, Columns: {len(raw_df.columns)}")
    
    knockout_stage_names = [
        'ROUND OF 32', 'ROUND OF 16', 'QUARTER FINALS', 'SEMI FINALS', 'FINAL'
    ]
    
    # Reshape Team 1 observations
    t1 = pd.DataFrame({
        'Date': raw_df['Date'],
        'Match_ID': raw_df.index + 1,
        'Stage': raw_df['Stage'],
        'Team': raw_df['Team 1'].str.strip(),
        'Opponent': raw_df['Team 2'].str.strip(),
        'Fouls': raw_df['Fouls_T1'].astype(float),
        'Yellow_Cards': raw_df['Yellow_Cards_T1'].astype(float),
        'Red_Cards': raw_df['Red_Cards_T1'].astype(float),
        'Score_For': raw_df['Score 1'].astype(str),
        'Score_Against': raw_df['Score 2'].astype(str),
        'xG': raw_df['xG_T1'].astype(float),
        'Shots': raw_df['Shots_T1'].astype(float),
        'Shots_On_Target': raw_df['Shots_On_Target_T1'].astype(float)
    })
    
    # Reshape Team 2 observations
    t2 = pd.DataFrame({
        'Date': raw_df['Date'],
        'Match_ID': raw_df.index + 1,
        'Stage': raw_df['Stage'],
        'Team': raw_df['Team 2'].str.strip(),
        'Opponent': raw_df['Team 1'].str.strip(),
        'Fouls': raw_df['Fouls_T2'].astype(float),
        'Yellow_Cards': raw_df['Yellow_Cards_T2'].astype(float),
        'Red_Cards': raw_df['Red_Cards_T2'].astype(float),
        'Score_For': raw_df['Score 2'].astype(str),
        'Score_Against': raw_df['Score 1'].astype(str),
        'xG': raw_df['xG_T2'].astype(float),
        'Shots': raw_df['Shots_T2'].astype(float),
        'Shots_On_Target': raw_df['Shots_On_Target_T2'].astype(float)
    })
    
    tidy_matches = pd.concat([t1, t2], ignore_index=True)
    
    # Classify stage type
    tidy_matches['Stage_Type'] = tidy_matches['Stage'].apply(
        lambda s: 'Knockout Stage' if s in knockout_stage_names else 'Group Stage'
    )
    
    # Identify teams that reached the Knockout Stage
    ko_teams = set(tidy_matches[tidy_matches['Stage_Type'] == 'Knockout Stage']['Team'].unique())
    all_teams = set(tidy_matches['Team'].unique())
    elim_teams = all_teams - ko_teams
    
    tidy_matches['Reached_Knockout'] = tidy_matches['Team'].isin(ko_teams)
    tidy_matches['Tournament_Status'] = tidy_matches['Reached_Knockout'].map({
        True: 'Knockout Stage',
        False: 'Eliminated in Group Stage'
    })
    
    # Sort logically
    tidy_matches = tidy_matches.sort_values(by=['Match_ID', 'Team']).reset_index(drop=True)
    
    # Save tidy match data
    out_tidy = os.path.join(PROCESSED_DIR, 'team_match_data.csv')
    tidy_matches.to_csv(out_tidy, index=False)
    print(f"Exported tidy team-match dataset to: {out_tidy}")
    print(f"Total observations: {len(tidy_matches)}")
    print(f"Knockout teams count (n1): {len(ko_teams)}")
    print(f"Group eliminated teams count (n2): {len(elim_teams)}")
    
    return tidy_matches, ko_teams, elim_teams


def prepare_and_sample(tidy_matches):
    """
    Aggregates metrics to the team level and defines comparison samples.
    """
    print("\n" + "="*70)
    print("2. DATA PREPARATION AND SAMPLING")
    print("="*70)
    
    # Team overall aggregations across all matches played
    team_overall = tidy_matches.groupby(['Team', 'Tournament_Status']).agg(
        Matches_Total=('Fouls', 'count'),
        Total_Fouls_Overall=('Fouls', 'sum'),
        Fouls_Per_Match_Overall=('Fouls', 'mean'),
        Fouls_Std_Overall=('Fouls', 'std')
    ).reset_index()
    
    # Team aggregations for Group Stage matches only
    gs_only = tidy_matches[tidy_matches['Stage'] == 'GROUP STAGE']
    team_gs = gs_only.groupby('Team').agg(
        Matches_GS=('Fouls', 'count'),
        Total_Fouls_GS=('Fouls', 'sum'),
        Fouls_Per_Match_GS=('Fouls', 'mean'),
        Fouls_Std_GS=('Fouls', 'std')
    ).reset_index()
    
    # Merge overall and group-stage summaries
    team_summary = pd.merge(team_overall, team_gs, on='Team', how='left')
    team_summary['Fouls_Std_Overall'] = team_summary['Fouls_Std_Overall'].fillna(0)
    team_summary['Fouls_Std_GS'] = team_summary['Fouls_Std_GS'].fillna(0)
    
    # Export team summary
    out_summary = os.path.join(PROCESSED_DIR, 'team_fouls_summary.csv')
    team_summary.to_csv(out_summary, index=False)
    print(f"Exported team summary dataset to: {out_summary}")
    print(f"Total teams aggregated: {len(team_summary)}")
    
    n_ko = (team_summary['Tournament_Status'] == 'Knockout Stage').sum()
    n_elim = (team_summary['Tournament_Status'] == 'Eliminated in Group Stage').sum()
    print(f"Sample 1 (Knockout Stage teams, n1): {n_ko}")
    print(f"Sample 2 (Eliminated in Group Stage teams, n2): {n_elim}")
    
    return team_summary


def main():
    print("Running preparedata.py...")
    tidy_matches, ko_teams, elim_teams = wrangle_data(RAW_DATA_PATH)
    team_summary = prepare_and_sample(tidy_matches)
    print("\nData preparation complete! Processed files are ready in data/processed/")


if __name__ == '__main__':
    main()
