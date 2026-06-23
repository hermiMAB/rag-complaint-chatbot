import pandas as pd
import re # For Regular Expressions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

import nltk

nltk.download('wordnet')



def load_data(file_path):
    """Loads a dataset from CSV and prints basic shape and memory information."""
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory : {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
    return df



def assess_and_clean_missing_data(df, critical_columns):
    """
    Assesses missing data for specific columns and safely drops rows missing critical info.
    """
    print("--- Missing/Empty Critical Data Report ---")
    
    # 1. Create a clean copy to avoid SettingWithCopyWarning
    df_working = df.copy()
    
    # 2. Check and print stats for ONLY the critical columns
    for col in critical_columns:
        nan_count = df_working[col].isnull().sum()
        empty_count = (df_working[col] == '').sum()
        print(f"{col} -> NaN count: {nan_count:,} | Empty string count: {empty_count:,}")
        
        # Convert empty strings to actual NaNs so dropna() will catch them
        df_working[col] = df_working[col].replace('', np.nan)
        
    print("-" * 50)
    print(f" Dropping rows with missing values in: {critical_columns}")
    
    # 3. Drop the rows
    df_cleaned = df_working.dropna(subset=critical_columns)
    
    # 4. Print the final outcome
    rows_dropped = len(df) - len(df_cleaned)
    print(f"\n--- Cleaning Results ---")
    print(f"Rows before cleaning : {len(df):,}")
    print(f"Rows after cleaning  : {len(df_cleaned):,}")
    print(f"Total rows dropped   : {rows_dropped:,}")
    
    return df_cleaned


def standardize_target_products(df,output_filepath):
    df_working = df.copy()
    # 3. Map the target products
    product_mapping = {
        'Credit card': 'Credit Card',
        'Credit card or prepaid card': 'Credit Card',
        'Checking or savings account': 'Savings Account',
        'Money transfer, virtual currency, or money service': 'Money Transfer',
        'Money transfers': 'Money Transfer',
        'Payday loan, title loan, or personal loan': 'Personal Loan',
        'Payday loan, title loan, personal loan, or advance loan': 'Personal Loan'
    }

    df_working['Product'] = df_working['Product'].map(product_mapping)
    print("distibution of products across the dataset")
    print (df_working['Product'].value_counts())
    # 4. Drop all the unmapped products (anything that became NaN)
    df_final = df_working.dropna(subset=['Product'])

    # 5. Save the final clean file
    df_final.to_csv(output_filepath, index=False)

    print("\n" + "="*45)
    print("🎯 FINAL EDA: PIPELINE ASSESSMENT")
    print("="*45)
    print(f"Final Usable Dataset Size : {len(df_final):,} rows")
    print("="*45)
    print("Saved successfully!")




def plot_categorical_distribution(df, categorical_col, title, palette='viridis', save_path=None):
    """
    Plots a generalized bar and donut chart for any categorical column.
    """
    # 1. Calculate general counts
    counts = df[categorical_col].value_counts()
    
    # 2. Print dynamic text summary
    summary = pd.DataFrame({
        "Count": counts.values,
        "Percentage (%)": (counts / counts.sum() * 100).round(2)
    }, index=counts.index)
    display(summary)
        
    # 3. Setup visualizations
    fig, axes = plt.subplots(1, figsize=(14, 5))
    colors = sns.color_palette(palette, len(counts))
    
    # --- Bar Chart ---
    bars = axes.bar(counts.index, counts.values, color=colors, edgecolor="white")
    axes.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=11)
    axes.set_title(f"Counts: {categorical_col}", fontweight="bold")
    axes.set_ylabel("Count")
    axes.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    
    # Rotate x-labels slightly so long product names don't overlap
    axes.tick_params(axis='x', rotation=15) 
    
    # --- Final Formatting ---
    plt.suptitle(title, fontsize=15, fontweight="bold", y=1.05)   
    plt.show()



import pandas as pd
import numpy as np

def analyze_narrative_lengths(df, text_column='Consumer complaint narrative', short_thresh=10, long_thresh=1000):
    """
    Creates a word count column and prints a statistical summary of extreme entries.
    """
    print(f"Analyzing narrative lengths for '{text_column}'...\n")
    
    # 1. CREATE THE COLUMN
    df['word_count'] = df[text_column].fillna('').astype(str).str.split().str.len()
    
    # 2. CALCULATE STATISTICS
    stats = df['word_count'].describe()
    very_short = df[df['word_count'] <= short_thresh]
    very_long = df[df['word_count'] >= long_thresh]
    
    # 3. PRINT SUMMARY
    print("="*45)
    print(" NARRATIVE LENGTH STATISTICAL SUMMARY")
    print("="*45)
    print(f"Total Records Analyzed : {int(stats['count']):,}")
    print(f"Average Word Count     : {stats['mean']:.1f} words")
    print(f"Median Word Count      : {int(stats['50%'])} words")
    print(f"Maximum Word Count     : {int(stats['max']):,} words")
    
    # Return the dataframe so the new 'word_count' column is saved
    return df

def plot_word_count_distribution(df, count_col='word_count', title='Distribution of Consumer Narrative Lengths'):
    """
    Plots a right-skew optimized histogram for text length data.
    """
    print(f"Generating distribution plot for '{count_col}'...\n")
    
    percentile_99 = df[count_col].quantile(0.99)
    median_val = df[count_col].median()
    
    plt.figure(figsize=(12, 6))
    
    # 1. Draw the histogram normally WITHOUT the log_scale parameter
    ax = sns.histplot(
        data=df, 
        x=count_col, 
        bins=60, 
        color='#4C72B0', 
        edgecolor='white'
    )
    
    # 2. Force the Y-axis into a log scale AFTER it is drawn
    plt.yscale('log')
    
    # Add the visual anchor for the median
    plt.axvline(median_val, color='#DD4444', linestyle='--', linewidth=2, label=f'Median: {int(median_val)} words')
    
    # Restrict the X-axis to the 99th percentile so outliers don't ruin the chart
    plt.xlim(0, percentile_99)
    
    # Formatting
    plt.title(title, fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Word Count (Capped at 99th Percentile)', fontsize=12)
    plt.ylabel('Frequency (Log Scale)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()