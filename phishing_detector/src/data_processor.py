import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from advanced_features import get_advanced_features

# Ensure you have the necessary NLTK data files
# Uncomment the next two lines and run them from a python interpreter to download the files:
# nltk.download('punkt')
# nltk.download('stopwords')

def clean_text(text):
    """
    Performs basic text cleaning:
    - Removes URLs
    - Removes punctuation and special characters
    - Converts text to lowercase
    - Removes numbers
    - Removes stopwords
    - Joins the processed tokens back into a string
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into a string
    return ' '.join(filtered_tokens)

def load_and_combine_data(data_dir):
    """
    Loads and combines multiple datasets from a specified directory,
    handling different column names and creating a unified 'text' column.

    Args:
        data_dir (str): The path to the directory containing the dataset files.

    Returns:
        pd.DataFrame: A single DataFrame containing all combined data.
        None: If no files are found or an error occurs.
    """
    all_data = []
    # List the files in the directory
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not files:
        print(f"Error: No CSV files found in directory: {data_dir}")
        return None
        
    print(f"Found {len(files)} files to combine: {files}")

    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        try:
            df = pd.read_csv(file_path, encoding='latin-1', on_bad_lines='skip')
            
            text_column = None
            label_column = None

            if 'body' in df.columns and 'label' in df.columns:
                text_column = 'body'
                label_column = 'label'
                if 'subject' in df.columns:
                    df['combined_text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
                    text_column = 'combined_text'
            
            elif 'text' in df.columns and 'label' in df.columns:
                text_column = 'text'
                label_column = 'label'
            
            elif 'Email' in df.columns and 'Spam' in df.columns:
                df.rename(columns={'Email': 'text', 'Spam': 'label'}, inplace=True)
                text_column = 'text'
                label_column = 'label'
            
            elif 'email_text' in df.columns and 'Email_type' in df.columns:
                 df.rename(columns={'email_text': 'text', 'Email_type': 'label'}, inplace=True)
                 text_column = 'text'
                 label_column = 'label'
            
            elif 'text_combined' in df.columns and 'label' in df.columns:
                text_column = 'text_combined'
                label_column = 'label'
            
            if text_column and label_column:
                temp_df = pd.DataFrame({
                    'text': df[text_column],
                    'label': df[label_column]
                })
                all_data.append(temp_df)
                print(f"Successfully loaded and added {file_name}.")
            else:
                print(f"Warning: Skipping {file_name} due to unexpected columns: {df.columns.tolist()}")
                continue
            
        except Exception as e:
            print(f"An error occurred while reading {file_name}: {e}")
            
    if not all_data:
        print("No valid datasets were loaded. Aborting.")
        return None

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def main():
    """
    Main function to execute the data loading, cleaning, and initial inspection.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(base_dir, "..", "data", "raw")

    print("Step 1: Loading and combining datasets...")
    df = load_and_combine_data(raw_data_dir)

    if df is not None:
        print("\nStep 2: Cleaning the text data and extracting advanced features...")
        # Apply basic text cleaning
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        # Extract ALL ten advanced features and convert to a list of dictionaries
        advanced_features_list = df['text'].apply(get_advanced_features).tolist()
        
        # Create a DataFrame directly from the list of dictionaries (guarantees all keys are columns)
        advanced_features_df = pd.DataFrame(advanced_features_list)
        
        feature_cols = [
       'num_urls', 'has_suspicious_keywords', 'has_suspicious_title', 
       'has_hidden_images', 'has_suspicious_url_pattern', 
       'has_high_html_ratio', 'has_common_typos', 'has_spoofed_header',
       'is_newly_registered_domain', 'has_typosquatting_link',
       'has_suspicious_tld', 'has_url_obfuscation', 'has_urgency_language', 'has_poor_grammar'
]
        # Defensive Coding: Check for and add missing columns, filling with 0
        for col in feature_cols:
            if col not in advanced_features_df.columns:
                advanced_features_df[col] = 0
        
        # Select only the required and correctly ordered columns
        df = pd.concat([df, advanced_features_df[feature_cols]], axis=1)

        processed_data_path = os.path.join(base_dir, "..", "data", "processed", "cleaned_dataset.csv")
        df.to_csv(processed_data_path, index=False, encoding='utf-8')
        print(f"\nCleaned dataset saved successfully to {processed_data_path}")

        print("\n-- Combined and Cleaned Dataset Information --")
        df.info()

        print("\n-- First 5 Rows of the Cleaned Dataset with new features --")
        print(df.head())
        
        print("\n-- Label Distribution --")
        print(df['label'].value_counts())

if __name__ == "__main__":
    main()
