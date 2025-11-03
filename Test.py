# Define a function to flip the data
def flip_data(df):
    # Create the winning team's perspective (Y = 1)
    win_df = df.copy()
    win_df['CurrentTeamID'] = win_df['WTeamID']
    win_df['OpponentTeamID'] = win_df['LTeamID']
    win_df['Result'] = 1  # Win

    # Create the losing team's perspective (Y = 0)
    lose_df = df.copy()
    lose_df['CurrentTeamID'] = lose_df['LTeamID']
    lose_df['OpponentTeamID'] = lose_df['WTeamID']
    lose_df['Result'] = 0  # Loss

    # Rename columns for the winning team's perspective
    win_df = win_df.rename(columns={
        'WFGM': 'CurrentTeam_FGM',
        'WFGA': 'CurrentTeam_FGA',
        'WFGM3': 'CurrentTeam_FGM3',
        'WFGA3': 'CurrentTeam_FGA3',
        'WFTM': 'CurrentTeam_FTM',
        'WFTA': 'CurrentTeam_FTA',
        'WOR': 'CurrentTeam_OR',
        'WDR': 'CurrentTeam_DR',
        'WAst': 'CurrentTeam_Ast',
        'WTO': 'CurrentTeam_TO',
        'WStl': 'CurrentTeam_Stl',
        'WBlk': 'CurrentTeam_Blk',
        'WPF': 'CurrentTeam_PF',
        'LFGM': 'OpponentTeam_FGM',
        'LFGA': 'OpponentTeam_FGA',
        'LFGM3': 'OpponentTeam_FGM3',
        'LFGA3': 'OpponentTeam_FGA3',
        'LFTM': 'OpponentTeam_FTM',
        'LFTA': 'OpponentTeam_FTA',
        'LOR': 'OpponentTeam_OR',
        'LDR': 'OpponentTeam_DR',
        'LAst': 'OpponentTeam_Ast',
        'LTO': 'OpponentTeam_TO',
        'LStl': 'OpponentTeam_Stl',
        'LBlk': 'OpponentTeam_Blk',
        'LPF': 'OpponentTeam_PF',
        'WScore': 'CurrentTeam_Score',
        'LScore': 'OpponentTeam_Score',
    })

    # Rename columns for the losing team's perspective
    lose_df = lose_df.rename(columns={
        'LFGM': 'CurrentTeam_FGM',
        'LFGA': 'CurrentTeam_FGA',
        'LFGM3': 'CurrentTeam_FGM3',
        'LFGA3': 'CurrentTeam_FGA3',
        'LFTM': 'CurrentTeam_FTM',
        'LFTA': 'CurrentTeam_FTA',
        'LOR': 'CurrentTeam_OR',
        'LDR': 'CurrentTeam_DR',
        'LAst': 'CurrentTeam_Ast',
        'LTO': 'CurrentTeam_TO',
        'LStl': 'CurrentTeam_Stl',
        'LBlk': 'CurrentTeam_Blk',
        'LPF': 'CurrentTeam_PF',
        'WFGM': 'OpponentTeam_FGM',
        'WFGA': 'OpponentTeam_FGA',
        'WFGM3': 'OpponentTeam_FGM3',
        'WFGA3': 'OpponentTeam_FGA3',
        'WFTM': 'OpponentTeam_FTM',
        'WFTA': 'OpponentTeam_FTA',
        'WOR': 'OpponentTeam_OR',
        'WDR': 'OpponentTeam_DR',
        'WAst': 'OpponentTeam_Ast',
        'WTO': 'OpponentTeam_TO',
        'WStl': 'OpponentTeam_Stl',
        'WBlk': 'OpponentTeam_Blk',
        'WPF': 'OpponentTeam_PF',
        'LScore': 'CurrentTeam_Score',
        'WScore': 'OpponentTeam_Score',
    })

    # Combine the two DataFrames
    flipped_df = pd.concat([win_df, lose_df], ignore_index=True)

    # Drop the original 'W' and 'L' columns
    flipped_df = flipped_df.drop(columns=['WTeamID', 'LTeamID', 'WScore', 'LScore'])

    return flipped_df

# Apply the flipping operation
flipped_df = flip_data(df)

# Display the resulting DataFrame
print(flipped_df)