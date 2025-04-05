import os
import re
from pathlib import Path

def process_folder(data_dir : Path, output_dir : str = "processed_files"):
    """
    Recursively process all text files in the given data directory.
    Replicates the structure and processes files into separate articles.
    """
    data_dir = Path(data_dir).resolve()
    base_output_dir = data_dir.parent / output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    
    for txt_file in data_dir.rglob("*.txt"):  # Recursively find all .txt files
        relative_path = txt_file.relative_to(data_dir)
        output_dir = base_output_dir / relative_path.parent / txt_file.stem
        os.makedirs(output_dir, exist_ok=True)
        process_file(txt_file, output_dir)
        
    print(f"Processing complete. Results stored in {base_output_dir}/")


def process_file(input_file, output_dir):
    """
    Process a single text file and save articles to the appropriate directory.
    """
    print(f"Processing {input_file}...")
    
    # Read the input file with error handling for encoding issues
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(input_file, "r", encoding="latin-1") as f:
            content = f.read()
    
    article_separator = r'(?:-{5,}|_{10,})'  # Match 5+ dashes OR 10+ underscores
    articles = re.split(article_separator, content)
    
    article_count = 0
    for article in articles:
        if not article.strip():
            continue
        
        cleaned_article = clean_article(article)
        if len(cleaned_article.split()) < 20:
            continue
        
        article_count += 1
        output_file = output_dir / f"article_{article_count}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(cleaned_article)
    
    print(f"  Extracted {article_count} articles to {output_dir}/")


def clean_article(text):
    """
    Clean an individual article by removing headers, standardizing format, and removing links.
    """
    text = re.sub(r'={10,}.*?={10,}', '', text, flags=re.DOTALL)  # Remove headers
    text = re.sub(r'\*{5,}', '', text)  # Remove asterisk sections
    text = re.sub(r'\s*\n\s*\n\s*', '\n\n', text)  # Normalize newlines
    text = re.sub(r'\s+', ' ', text).strip()  # Remove excess whitespace
    # Remove links (URLs)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    return text.strip()

if __name__ == "__main__":
    data_folder = Path.cwd().parent / "data"
    process_folder(data_folder)