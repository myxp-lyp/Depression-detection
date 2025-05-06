import os
import pandas as pd
import csv
import numpy as np

# Define the base directory
base_dir = r"D:\Files\DAIC-WOZ"

# Define the folders to process (excluding the specified ones)
excluded_folders = ['342', '394', '398', '460']
folder_range = range(300, 493)
folders_to_process = [str(folder_id) for folder_id in folder_range if str(folder_id) not in excluded_folders]

# Initialize the output DataFrame
output_data = []

# Process each folder
for folder_id in folders_to_process:
    folder_path = os.path.join(base_dir, folder_id)
    
    # Skip if the folder doesn't exist
    if not os.path.exists(folder_path):
        continue
    
    # Find the transcript file
    transcript_file = os.path.join(folder_path, f"{folder_id}_TRANSCRIPT.csv")
    
    # Skip if the transcript file doesn't exist
    if not os.path.exists(transcript_file):
        continue
    
    try:
        # Read the transcript file
        # Using tab as separator and handling potential quoting issues
        transcript_df = pd.read_csv(transcript_file, sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8')
        
        # Filter for "Participant" rows and extract values
        participant_values = transcript_df[transcript_df['speaker'] == 'Participant']['value'].tolist()
        
        # Convert any NaN values to empty strings
        participant_values = [str(val) if not pd.isna(val) else "" for val in participant_values]
        
        # Filter out empty values
        participant_values = [val for val in participant_values if val.strip()]
        
        # Add period to each value if it doesn't already end with one
        participant_values = [
            val + "." if not val.endswith((".","?","!")) else val 
            for val in participant_values
        ]
        
        # Combine every 5 utterances
        paragraphs = []
        for i in range(0, len(participant_values), 5):
            paragraph = " ".join(participant_values[i:i+5])
            if paragraph:  # Only add if not empty
                paragraphs.append(paragraph)
        
        # Add to output data
        for paragraph in paragraphs:
            output_data.append({
                'id': folder_id,
                'index': paragraph
            })
            
    except Exception as e:
        print(f"Error processing folder {folder_id}: {e}")

# Create the output DataFrame
output_df = pd.DataFrame(output_data)

# Save to processed_transcript.csv
output_file = os.path.join(base_dir, "processed_transcript.csv")
output_df.to_csv(output_file, index=False)

print(f"Processing complete. Output saved to {output_file}")