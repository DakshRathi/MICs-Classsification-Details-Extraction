import os
import csv
from pathlib import Path
from typing import Optional
from tqdm.auto import tqdm
import argparse
import sys
from torch.utils.data import DataLoader

from utils import init_llm, ArticleDataset, get_device
from preprocessing import process_folder
from classification_chain import create_classification_chain
from extraction_chain import create_extraction_chain


def process_batch(batch, classification_chain, extraction_chain):
    """
    Process a batch of articles through classification and extraction.
    
    Args:
        batch (dict): Batch of articles with file_index and content keys
        classification_chain: Chain for classifying articles
        extraction_chain: Chain for extracting details from MIC articles
        
    Returns:
        list: Results containing classification and extraction outcomes
    """
    results = []
    
    for i in range(len(batch["file_index"])):
        file_index = batch["file_index"][i]
        content = batch["content"][i]
        
        if not content:
            print(f"Skipping empty content for {file_index}")
            continue
        
        try:
            # Classify the article
            classification_result = classification_chain.invoke({"article": content})
            
            # If classified as MIC, extract details immediately
            extraction_result = None
            if classification_result.is_mic:
                extraction_result = extraction_chain.invoke({"article": content})
            
            results.append({
                "file_index": file_index,
                "classification_result": classification_result,
                "extraction_result": extraction_result,
                "is_mic": classification_result.is_mic
            })
        except Exception as e:
            print(f"Error processing {file_index}: {str(e)[:200]}...")
            # Continue processing other files in the batch
        
    return results


def save_results(results, classification_file, extraction_file):
    """
    Save batch results to classification and extraction CSV files.
    
    Args:
        results (list): List of processed article results
        classification_file (Path): Path to classification output file
        extraction_file (Path): Path to extraction output file
        
    Returns:
        tuple: Count of new classifications and extractions
    """
    # Check if files exist to determine if headers are needed
    classification_exists = classification_file.exists()
    extraction_exists = extraction_file.exists()
    
    # Separate classification and extraction results
    classifications = []
    extractions = []
    
    for result in results:
        if result["classification_result"]:
            classifications.append((result["file_index"], result["classification_result"]))
        
        if result["extraction_result"]:
            extractions.append((result["file_index"], result["extraction_result"]))
    
    # Save classification results
    if classifications:
        with open(classification_file, mode="a" if classification_exists else "w", 
                 newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not classification_exists:
                writer.writerow(["Index", "Label", "Explanation"])
            
            for file_index, result in classifications:
                writer.writerow([
                    file_index,
                    int(result.is_mic),
                    result.explanation
                ])
    
    # Save extraction results
    if extractions:
        with open(extraction_file, mode="a" if extraction_exists else "w", 
                 newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not extraction_exists:
                writer.writerow([
                    'Index', 'MICdate', 'Fatality_Min', 'Fatality_Max',
                    'Countries_Involved', 'Initiator_Country', 'Target_Country'
                ])
            
            for file_index, result in extractions:
                writer.writerow([
                    file_index,
                    result.MICdate,
                    result.fatality_min,
                    result.fatality_max,
                    ', '.join(result.countries_involved) if result.countries_involved else '',
                    result.initiator_country if result.initiator_country else "null",
                    result.target_country if result.target_country else "null"
                ])
    
    return len(classifications), len(extractions)


def custom_collate_fn(batch):
    """
    Custom collate function to handle None values in batches.
    
    Args:
        batch (list): List of dictionaries with file_index and content
        
    Returns:
        dict: Filtered batch with valid entries
    """
    valid_batch = [item for item in batch if item["content"] is not None]
    if not valid_batch:
        return {"file_index": [], "content": []}
    
    return {
        "file_index": [item["file_index"] for item in valid_batch],
        "content": [item["content"] for item in valid_batch]
    }


def process_directory(
    current_dir: Path,
    base_dir: Path,
    output_dir: Path,
    classification_chain,
    extraction_chain,
    batch_size: int = 4
):
    """
    Process a single directory with unified classification and extraction.
    
    Args:
        current_dir (Path): Directory containing text files to process
        base_dir (Path): Root directory for calculating relative paths
        output_dir (Path): Base output directory
        classification_chain: Chain for classification
        extraction_chain: Chain for extraction
        batch_size (int): Batch size for processing
    
    Returns:
        tuple: Count of processed articles and identified MICs
    """
    # Find all text files in the current directory
    article_files = sorted(current_dir.glob("*.txt"), key=lambda x: x.name)
    
    if not article_files:
        return 0, 0
    
    # Calculate relative path from base directory
    try:
        relative_path = current_dir.relative_to(base_dir)
    except ValueError:
        # This is the base directory itself
        relative_path = Path("")
    
    # Create output directories
    classification_dir = output_dir / "classified_files" / relative_path
    extraction_dir = output_dir / "detailed_files" / relative_path
    os.makedirs(classification_dir, exist_ok=True)
    os.makedirs(extraction_dir, exist_ok=True)
    
    # Create output files
    classification_file = classification_dir / "classification.csv"
    extraction_file = extraction_dir / "details.csv"
    
    # Get existing processed files
    existing_entries = set()
    if classification_file.exists():
        with open(classification_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if row:
                    existing_entries.add(row[0])
    
    # Filter out already processed files
    unprocessed_files = [f for f in article_files if f.name not in existing_entries]
    
    if not unprocessed_files:
        print(f"Skipping {current_dir}, all files already processed.")
        return 0, 0
    
    print(f"\nProcessing directory: {current_dir}")
    print(f" - Found {len(article_files)} total articles")
    print(f" - Processing {len(unprocessed_files)} new articles")
    
    # Create dataset and dataloader with parallel workers
    dataset = ArticleDataset(unprocessed_files)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if get_device() == "cuda" or get_device() == "mps" else False,
        collate_fn=custom_collate_fn
    )
    
    # Process articles in batches
    total_classified = 0
    total_extracted = 0
    
    with tqdm(total=len(unprocessed_files), desc=f"Processing {relative_path}", leave=False) as pbar:
        for batch in dataloader:
            # Skip empty batches
            if not batch["file_index"]:
                continue
            
            # Process the batch
            results = process_batch(batch, classification_chain, extraction_chain)
            
            # Save results
            new_classifications, new_extractions = save_results(
                results,
                classification_file,
                extraction_file
            )
            
            total_classified += new_classifications
            total_extracted += new_extractions
            
            # Update progress bar
            pbar.update(len(batch["file_index"]))
            pbar.set_postfix({
                "Classified": total_classified,
                "MICs": total_extracted
            })
    
    print(f"✅ Completed directory: {current_dir}")
    print(f" - New classifications: {total_classified}")
    print(f" - New MICs identified: {total_extracted}")
    
    return total_classified, total_extracted


def run_unified_pipeline(
    raw_data_dir: Path, 
    output_dir: Path, 
    preprocess: bool = True,
    hf_model_name: Optional[str] = None,
    batch_size: int = 4,
):
    """
    End-to-end pipeline for preprocessing, classification, and extraction.
    
    Args:
        raw_data_dir (Path): Directory containing raw text files
        output_dir (Path): Directory to store processed files and results
        preprocess (bool): Whether to preprocess the data
        hf_model_name (str, optional): Hugging Face model name
        batch_size (int): Number of articles to process simultaneously
    """
    # Resolve paths
    raw_data_dir = Path(raw_data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    processed_dir = output_dir / "processed_files"
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Preprocessing (if enabled)
    if preprocess:
        print("\n" + "="*50)
        print("Starting Preprocessing Stage".center(50))
        print("="*50)
        # Call the preprocessing function
        process_folder(raw_data_dir, processed_dir)
    
    # Step 2: Classification & Extraction
    print("\n" + "="*50)
    print("Starting Analysis Stage".center(50))
    print("="*50)
    
    # Initialize model chains
    llm = init_llm(hf_model_name)
    classification_chain = create_classification_chain(llm)
    extraction_chain = create_extraction_chain(llm)

    # Track total statistics
    total_articles = 0
    total_mic = 0
    
    # Process the root directory first (for files directly in processed_dir)
    root_articles, root_mic = process_directory(
        processed_dir,
        processed_dir,
        output_dir,
        classification_chain,
        extraction_chain,
        batch_size
    )
    
    total_articles += root_articles
    total_mic += root_mic
    
    # Process all subdirectories recursively
    for directory in sorted(processed_dir.rglob("*")):
        if not directory.is_dir() or directory == processed_dir:
            continue
            
        # Process the directory
        dir_articles, dir_mic = process_directory(
            directory,
            processed_dir,
            output_dir,
            classification_chain,
            extraction_chain,
            batch_size
        )
        
        total_articles += dir_articles
        total_mic += dir_mic
    
    print("\n" + "="*50)
    print("Processing Complete".center(50))
    print("="*50)
    print(f"Total articles processed: {total_articles}")
    print(f"Total MICs identified: {total_mic}")


def main():
    """Command-line interface for the MIC analysis pipeline"""
    parser = argparse.ArgumentParser(
        description="Militarized Interstate Confrontation Analysis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "raw_data_dir",
        type=str,
        help="Path to directory containing raw text files"
    )
    
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to directory for storing processed results"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Hugging Face model name for classification/extraction"
    )
    
    parser.add_argument(
        "--no-preprocess", 
        action="store_false",
        dest="preprocess",
        help="Skip preprocessing step"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of articles to process simultaneously"
    )
    
    try:
        args = parser.parse_args()
    except SystemExit:
        print("\nError: Missing required arguments. See usage below:\n")
        parser.print_help()
        sys.exit(1)
    
    # Convert paths to Path objects
    raw_data_path = Path(args.raw_data_dir).resolve()
    output_path = Path(args.output_dir).resolve()
    
    # Input validation
    if not raw_data_path.exists():
        print(f"Error: Input directory not found: {raw_data_path}")
        sys.exit(1)
        
    if not raw_data_path.is_dir():
        print(f"Error: Input path is not a directory: {raw_data_path}")
        sys.exit(1)
    
    # Execute pipeline
    run_unified_pipeline(
        raw_data_dir=raw_data_path,
        output_dir=output_path,
        preprocess=args.preprocess,
        hf_model_name=args.model,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
