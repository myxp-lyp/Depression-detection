import pandas as pd
import os

# Define the paths to your files
base_dir = r"D:\Files\DAIC-WOZ"
processed_transcript_path = os.path.join(base_dir, "processed_transcript.csv")
phq_table_path = os.path.join(base_dir, "phq_table.csv")

# Load the processed transcript data
transcript_df = pd.read_csv(processed_transcript_path)

# Load the PHQ table data (comma-separated)
phq_df = pd.read_csv(phq_table_path)

# Create a mapping of Participant_ID to PHQ8_Score
phq_mapping = dict(zip(phq_df['Participant_ID'], phq_df['PHQ8_Score']))

# Add the PHQ scores to the transcript dataframe
transcript_df['phq_score'] = transcript_df['id'].map(phq_mapping)

# Save the updated dataframe
transcript_df.to_csv(processed_transcript_path, index=False)

print(f"Added PHQ scores to {processed_transcript_path}")