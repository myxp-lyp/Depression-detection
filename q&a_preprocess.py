import csv
import json
import requests
import re
import time
from pathlib import Path

# Configuration
INPUT_FILE = "complete_data.csv"
OUTPUT_FILE = "processed_data.csv"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"  # Update with actual API endpoint
API_KEY = ""  # Replace with your actual API key

# Define your questions here
questions = [
    "Construct 10 Q&A sets based on one [paragraph] I give you. The Q&A sets should be about the key definitions mentioned in the paragraph. The Q&A set should contain no extra knowledge. The entire conversation should contain English only. The message you reply must follow the exact format in the [example], do not add any extra \" or other marks at the begining or the end of your question or answer.\
    [example]:\
    question: This is the first question you construct.\
    answer: This is the first answer you construct.\
    question: This is the second question you construct.\
    answer: This is the second answer you construct.",

    "Construct 10 Q&A sets based on one [paragraph] I give you. The questions in these sets should be 'why' questions. The Q&A set should contain no extra knowledge. The entire conversation should contain English only. The message you reply must follow the exact format in the [example], do not add any extra \" or other marks at the begining or the end of your question or answer.\
    [example]:\
    question: This is the first question you construct.\
    answer: This is the first answer you construct.\
    question: This is the second question you construct.\
    answer: This is the second answer you construct.",

    "Construct 10 Q&A sets based on one [paragraph] I give you. The Q&A are about the phenomena that may occur on a people with such disorder. These phenomena should mimic people's spoken texts. I will provide you with one example. The Q&A set should contain no extra knowledge. The entire conversation should contain English only. The message you reply must follow the exact format in the [example], do not add any extra \" or other marks at the begining or the end of your question or answer.\
    [example]:\
    question: How does a people with depression talk like when you ask about their daily routine?\
    answer: My daily routine is quite simple, there's not much changes during my everyday life, I barely communicate with my family, and doesn't have much friends to hang out during free time.\
    question: This is the second question you construct.\
    answer: This is the second answer you construct.",

    "Construct 5 Q&A sets completely based on extended knowledge which are not mentioned in the [paragraph], but should also be considered important about such disorder. The entire conversation should contain English only. The message you reply must follow the exact format in the [example], do not add any extra \" or other marks at the begining or the end of your question or answer.\
    [example]:\
    question: This is the first question you construct.\
    answer: This is the first answer you construct.\
    question: This is the second question you construct.\
    answer: This is the second answer you construct.",
    
    "Based on the [paragraph], provide 5 Q&A sets with critical thinking. The Q&A set should contain no extra knowledge. The entire conversation should contain English only. The message you reply must follow the exact format in the [example], do not add any extra \" or other marks at the begining or the end of your question or answer.\
    [example]:\
    question: This is the first question you construct.\
    answer: This is the first answer you construct.\
    question: This is the second question you construct.\
    answer: This is the second answer you construct."
]

def setup_output_file():
    """Set up the output CSV file if it doesn't exist."""
    if not Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['question', 'answer'])

def call_deepseek_api(prompt):
    """Call the Deepseek-chat API with the given prompt."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": "deepseek-chat",  # Update with the correct model name
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing API response: {e}")
        return None

def extract_qa_pairs(response_text):
    """Extract question-answer pairs from the API response."""
    qa_pairs = []
    
    # Pattern to match question-answer pairs
    pattern = r'question:\s*(.*?)\s*answer:\s*(.*?)(?=\nquestion:|$)'
    
    # Find all matches in the response text
    matches = re.findall(pattern, response_text, re.DOTALL)
    
    for question, answer in matches:
        qa_pairs.append({
            'question': question.strip(),
            'answer': answer.strip()
        })
    
    return qa_pairs

def append_to_csv(qa_pairs):
    """Append the question-answer pairs to the output CSV file."""
    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        for qa in qa_pairs:
            writer.writerow([
                qa['question'],
                qa['answer']
            ])

def process_csv():
    """Process the input CSV file and call the Deepseek-chat API for each row."""
    # Ensure output file exists
    setup_output_file()
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            # Use csv.reader with appropriate settings to handle quoted fields
            reader = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
            
            # Skip header row if it exists
            header = next(reader, None)
            
            for row in reader:
                if len(row) < 2:
                    print(f"Warning: Skipping incomplete row: {row}")
                    continue
                
                title = row[0]
                description = row[1]
                
                print(f"Processing row with title: {title[:30]}...")
                
                # Process each question for this row
                for question in questions:
                    # Construct the prompt according to the specified format
                    prompt = f"[paragraph]:\nTitle: \"{title}\"\ncontext: \"{description}\"\n{question}"
                    
                    # Call the API
                    response = call_deepseek_api(prompt)

                    # print(response)  #debug
                    
                    if response:
                        # Extract question-answer pairs
                        qa_pairs = extract_qa_pairs(response)
                        
                        # Append to CSV
                        append_to_csv(qa_pairs)
                        
                        print(f"  - Processed question: {question[:30]}... ({len(qa_pairs)} QA pairs extracted)")
                    else:
                        print(f"  - Failed to process question: {question[:30]}...")
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(1)
    
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def add_quotation_marks():
    """Add quotation marks to elements in processed_data.csv if they don't already have them."""
    temp_file = "temp_processed_data.csv"
    
    try:
        # Read the existing data
        rows = []
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows.append(header)
            
            for row in reader:
                processed_row = []
                for item in row:
                    # Add quotation marks if the item doesn't start and end with them
                    if not (item.startswith('"') and item.endswith('"')):
                        item = f'"{item}"'
                    processed_row.append(item)
                rows.append(processed_row)
        
        # Write to a temporary file
        with open(temp_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        # Replace the original file
        Path(temp_file).replace(OUTPUT_FILE)
        print(f"Added quotation marks to elements in {OUTPUT_FILE}.")
    
    except Exception as e:
        print(f"Error adding quotation marks: {e}")
        if Path(temp_file).exists():
            Path(temp_file).unlink()
    
    except Exception as e:
        print(f"Error fixing quotation marks: {e}")
        if Path(temp_file).exists():
            Path(temp_file).unlink()

def delete_quotation_marks():
    # 打开文件并读取内容
    with open('processed_data.csv', 'r', encoding='utf-8') as file:
        content = file.read()

    # 替换所有的 """ 为 "
    content = content.replace('"""', '"')

    # 将修改后的内容写回文件
    with open('processed_data.csv', 'w', encoding='utf-8') as file:
        file.write(content)

if __name__ == "__main__":
    print(f"Starting to process {INPUT_FILE}...")
    # process_csv()
    add_quotation_marks()
    delete_quotation_marks()
    print(f"Processing complete. Results saved to {OUTPUT_FILE}.")