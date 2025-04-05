import argparse
from pathlib import Path
import sys

def main():
    """Command-line interface for the unified MIC analysis pipeline"""
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
    
    # Import pipeline function
    try:
        from unified_pipeline import run_unified_pipeline
    except ImportError:
        print("Error: Could not import pipeline components")
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
