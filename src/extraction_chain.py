import os
import csv
import pandas as pd
from datetime import date
from pathlib import Path
from typing import List, Optional, Union
from tqdm.auto import tqdm

from torch.utils.data import DataLoader

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from utils import init_llm, ArticleDataset

# Define a structured output model for the detailed MIC information
class MICDetailedInfo(BaseModel):
    """Schema for detailed information about a Militarized Interstate Confrontation."""
    MICdate: date = Field(description="The date when the MIC occurred (YYYY-MM-DD format, or as specific as possible)")
    fatality_min: int = Field(description="The minimum number of fatalities (use same number as max if precise)")
    fatality_max: int = Field(description="The maximum number of fatalities (use same number as min if precise)")
    countries_involved: List[str] = Field(description="List of countries involved in the confrontation")
    initiator_country: Optional[str] = Field(description="The country that initiated the confrontation, if identifiable")
    target_country: Optional[str] = Field(description="The country that was targeted in the confrontation, if identifiable")


def create_extraction_chain(llm: Union[HuggingFacePipeline, ChatOpenAI]) -> Runnable:
    """Create a LangChain chain for extracting detailed MIC information."""
    # Initialize components
    parser = PydanticOutputParser(pydantic_object=MICDetailedInfo)
    format_instructions = parser.get_format_instructions()

    prompt = PromptTemplate(
        template="""You are an expert analyst of international relations and military conflicts.

        This article has been identified as describing a Militarized Interstate Confrontation (MIC).

        Extract the following specific details about this confrontation:
        1. The date when the confrontation occurred (be as precise as possible, use YYYY-MM-DD format if date is known else return 0000-00-00)
        2. The number of fatalities (provide a range with minimum and maximum values; use the same number for both if precise)
        3. All countries involved in the confrontation
        4. If possible, identify which country initiated the confrontation and which was the target

        If any information is not explicitly stated in the article, make your best estimate based on context clues.
        If you cannot determine a piece of information at all, use null for that field.

        Example 1:
        Article: "On March 15, 2022, tensions escalated between Nation A and Nation B, leading to armed skirmishes. Reports confirm at least 50 casualties."

        Extracted Details:
        ```json
        {{
            "MICdate": "2022-03-15",
            "fatality_min": 50,
            "fatality_max": 50,
            "countries_involved": ["Nation A", "Nation B"],
            "initiator_country": "Nation A",
            "target_country": "Nation B"
        }}
        ```

        Example 2:
        Article: "In early 1998, a naval conflict arose between Country X and Country Y. The exact number of casualties remains unknown."

        Extracted Details:
        ```
        {{
            "MICdate": "1998-01-01",
            "fatality_min": 0,
            "fatality_max": 0,
            "countries_involved": ["Country X", "Country Y"],
            "initiator_country": null,
            "target_country": null
        }}
        ```

        Article:
        {article}

        {format_instructions}
        """,
        input_variables=['article'],
        partial_variables={'format_instructions': format_instructions}
    )

    # Build the LCEL chain
    chain = prompt | llm | parser

    return chain

def extract_mic_details(chain, article: str) -> MICDetailedInfo:
    try:
        return chain.invoke({"article": article})
    except Exception:
        return MICDetailedInfo(
            MICdate="0000-00-00",
            fatality_min=0,
            fatality_max=0,
            countries_involved=[],
            initiator_country=None,
            target_country=None
        )
    
def process_extraction_chain(processed_base_dir: Path, classified_base_dir: Path, detailed_base_dir: Path, hf_model_name: str = None) -> None:
    """
    Process MIC-classified articles to extract detailed information using recursive file logic
    and dataset/dataloader for efficient processing.

    Args:
        processed_base_dir (Path): Base directory containing processed article files
        classified_base_dir (Path): Base directory containing classification CSV files
        detailed_base_dir (Path): Base directory where detailed information CSVs will be saved
        hf_model_name (str, optional): Name of the Hugging Face model to use for extraction
    """
    # Initialize LLM and extraction chain
    llm = init_llm(hf_model_name)
    extraction_chain = create_extraction_chain(llm)

    # Ensure output directory exists
    processed_base_dir = processed_base_dir.resolve()
    classified_base_dir = classified_base_dir.resolve()
    detailed_base_dir = detailed_base_dir.resolve()
    os.makedirs(detailed_base_dir, exist_ok=True)

    print(f"Starting MIC detail extraction from: {classified_base_dir}")
    print(f"Using article content from: {processed_base_dir}")
    print(f"Saving detailed information to: {detailed_base_dir}")

    # Recursively process all classification CSV files
    for csv_file in sorted(classified_base_dir.rglob("*.csv")):
        # Get relative path from base directory
        relative_path = csv_file.parent.relative_to(classified_base_dir)

        # Determine folder name (will typically be year or other category)
        folder_name = csv_file.parent.name

        # Construct corresponding output folder and file
        output_folder = detailed_base_dir / relative_path
        os.makedirs(output_folder, exist_ok=True)
        output_file = output_folder / "details.csv"

        print(f"\nProcessing classification file: {csv_file}")

        # Read and filter MIC articles
        try:
            df = pd.read_csv(csv_file)
            mic_articles = df[df['Label'] == 1]

            if mic_articles.empty:
                print(f"No MIC articles found in {csv_file}. Skipping.")
                continue

            print(f"Found {len(mic_articles)} MIC articles in {folder_name}.")
        except Exception as e:
            print(f"Error reading classification file {csv_file}: {e}")
            continue

        # Load existing processed entries to avoid duplication
        existing_entries = set()
        if output_file.exists():
            with open(output_file, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if row:
                        existing_entries.add(row[0])  # Store Index values

            print(f"Found {len(existing_entries)} existing entries in {output_file}")

            # Filter out already processed entries
            unprocessed_articles = mic_articles[~mic_articles['Index'].isin(existing_entries)]
            if unprocessed_articles.empty:
                print(f"All MIC articles in {csv_file} already processed. Skipping.")
                continue

            print(f"{len(unprocessed_articles)} new MIC articles to process.")
        else:
            unprocessed_articles = mic_articles
            
        # Create a list of Path objects from the article indices
        article_paths = []
        for idx, row in unprocessed_articles.iterrows():
            article_name = row['Index']  # Get filename from Index column
            article_path = processed_base_dir / relative_path / article_name
            article_paths.append(article_path)

        # Create dataset and dataloader for efficient processing
        dataset = ArticleDataset(article_paths)
        
        # Define a custom collate function to handle None values
        def collate_fn(batch):
            # Filter out None values
            filtered_batch = [item for item in batch if item["content"] is not None]
            if not filtered_batch:
                return {"file_index": [], "content": []}
            
            # Manually collate valid items
            file_indices = [item["file_index"] for item in filtered_batch]
            contents = [item["content"] for item in filtered_batch]
            return {"file_index": file_indices, "content": contents}
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,  # Process 4 articles at a time
            shuffle=False,
            num_workers=0,  # Reduced worker count to avoid issues
            pin_memory=True, 
            collate_fn=collate_fn  # Use custom collate function
        )

        # Open output file in appropriate mode
        file_mode = "a" if output_file.exists() else "w"
        with open(output_file, mode=file_mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write headers if it's a new file
            if file_mode == "w":
                writer.writerow([
                    'Index', 'MICdate', 'Fatality_Min', 'Fatality_Max',
                    'Countries_Involved', 'Initiator_Country', 'Target_Country'
                ])

            # Process articles in batches
            for batch in tqdm(dataloader, desc=f"Extracting details from {folder_name}", leave=False):
                # Skip empty batches
                if not batch["file_index"]:
                    continue
                    
                for i in range(len(batch["file_index"])):
                    file_index = batch["file_index"][i]
                    content = batch["content"][i]

                    # Skip if content is empty
                    if not content:
                        continue

                    # Extract details
                    details = extract_mic_details(extraction_chain, content)

                    # Write extracted details to CSV
                    writer.writerow([
                        file_index,
                        details.MICdate,
                        details.fatality_min,
                        details.fatality_max,
                        ', '.join(details.countries_involved) if details.countries_involved else '',
                        details.initiator_country if details.initiator_country else "null",
                        details.target_country if details.target_country else "null"
                    ])

                # Flush file after each batch
                f.flush()

        print(f"✅ Finished processing: {csv_file} -> Output: {output_file}\n")


if __name__ == "__main__":
    processed_base_dir = Path.cwd().parent / "processed_files"
    classified_base_dir = Path.cwd().parent / "classified_files"
    detailed_base_dir = Path.cwd().parent / "detailed_files"
    hf_model_name = "Qwen/Qwen2.5-3B-Instruct"
    process_extraction_chain(processed_base_dir, classified_base_dir, detailed_base_dir, hf_model_name)
    print("Extraction and saving complete.")