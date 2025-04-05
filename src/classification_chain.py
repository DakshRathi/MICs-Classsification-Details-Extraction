import os
import csv
from pathlib import Path
from typing import Union

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from utils import init_llm, ArticleDataset

# Define the output model using Pydantic
class MICClassificationResult(BaseModel):
    """Schema for the Militarized Interstate Confrontation classification result."""
    is_mic: bool = Field(description="True if the article describes a Militarized Interstate Confrontation (MIC), False otherwise")
    explanation: str = Field(description="A brief explanation of why the article was classified as MIC or not")


def create_classification_chain(llm: Union[HuggingFacePipeline, ChatOpenAI]) -> Runnable:
    """Create a LangChain classification chain"""
    # Initialize components
    parser = PydanticOutputParser(pydantic_object=MICClassificationResult)
    format_instructions = parser.get_format_instructions()

    prompt = PromptTemplate(
        template="""You are an expert analyst of international relations and military conflicts.
        Your task is to determine whether a news article describes a Militarized Interstate Confrontation (MIC).

        A Militarized Interstate Confrontation (MIC) is defined as:
        - A direct confrontation between two or more countries
        - Involving military forces (army, navy, air force, etc.)
        - Where there is a threat, display, or use of military force

        The article must describe an actual military interaction, not just diplomatic tensions or discussions about potential conflicts.

        Analyze the following article carefully and determine if it describes a MIC.
        Output your answer in the specified JSON format with two fields:
        1. is_mic: true if it's a MIC, false if it's not
        2. explanation: Very short explanation of your reasoning

        Here are some examples to guide you:
        Example 1:
        Article: Russian troops opened fire on Ukrainian soldiers near the border, killing three and wounding seven others. The Ukrainian government condemned the attack as a violation of its sovereignty.
        Output: {{"is_mic": true, "explanation": "This article describes a direct military confrontation between Russian and Ukrainian forces with fatalities, which is a clear case of a Militarized Interstate Confrontation."}}

        Example 2:
        Article: China and Taiwan held diplomatic talks aimed at easing tensions in the region. Both sides agreed to maintain open lines of communication to prevent misunderstandings.
        Output: {{"is_mic": false, "explanation": "This article describes diplomatic talks rather than a military confrontation. No military forces were involved, and there was no threat or use of force."}}

        Example 3:
        Article: North Korean forces fired artillery shells into South Korean waters as a show of force during joint US-South Korean military exercises. No casualties were reported.
        Output: {{"is_mic": true, "explanation": "This article describes a militarized action (artillery fire) by North Korea directed at South Korea, which constitutes a Militarized Interstate Confrontation even without casualties."}}

        Example 4:
        Article: The United Nations Security Council met to discuss increasing tensions between India and Pakistan but no military actions were reported.
        Output: {{"is_mic": false, "explanation": "This article only mentions diplomatic discussions about tensions. It does not describe any actual military confrontation, threat, or use of force between countries."}}

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

def classify_article(chain, article_text: str) -> MICClassificationResult:
    """Classify an article as MIC or non-MIC using the given chain."""
    try:
        return chain.invoke({"article": article_text})
    except Exception as e:
        # Return a default response if parsing fails
        return MICClassificationResult(is_mic=False, explanation=f"Error in classification")
    

def process_classification_chain(processed_base_dir: Path, classified_base_dir: Path, hf_model_name: str = None) -> None:
    """
    Process articles from the processed folder structure and create corresponding
    CSV files with classification results, preserving the original directory structure.

    Args:
        processed_base_dir (Path): Directory containing processed article folders.
        classified_base_dir (Path): Directory where classification CSVs will be saved.
        hf_model_name (str): Name of the Hugging Face model to use for classification.
    """
    llm = init_llm(hf_model_name)
    chain = create_classification_chain(llm)

    processed_base_dir = processed_base_dir.resolve()
    classified_base_dir = classified_base_dir.resolve()
    os.makedirs(classified_base_dir, exist_ok=True)

    print(f"Starting classification from: {processed_base_dir}")
    print(f"Saving classification results to: {classified_base_dir}")

    # Process all directories, including root directory
    directories_to_process = [processed_base_dir]  # Start with root directory
    
    # Add all subdirectories
    for folder in sorted(processed_base_dir.rglob("*")):
        if folder.is_dir():
            directories_to_process.append(folder)
    
    # Process each directory
    for folder in directories_to_process:
        article_files = sorted(folder.glob("*.txt"), key=lambda x: x.name)
        if not article_files:
            continue

        # Determine the relative path (empty for root directory)
        try:
            relative_path = folder.relative_to(processed_base_dir)
        except ValueError:  # This is the root directory
            relative_path = Path("")
        
        output_folder = classified_base_dir / relative_path
        os.makedirs(output_folder, exist_ok=True)
        csv_file = output_folder / "classification.csv"

        # Load existing classified file indexes
        existing_entries = set()
        if csv_file.exists():
            with open(csv_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if row:
                        existing_entries.add(row[0])

        # Filter out already classified files
        unclassified_files = [
            file for file in article_files
            if f"{file.name}" not in existing_entries
        ]

        if not unclassified_files:
            print(f"Skipping {folder}, all files already classified.")
            continue

        print(f"\nProcessing folder: {folder}")
        print(f" - Found {len(article_files)} total articles.")
        print(f" - {len(unclassified_files)} unclassified articles to process.")

        dataset = ArticleDataset(unclassified_files)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

        file_mode = "a" if csv_file.exists() else "w"
        with open(csv_file, mode=file_mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if file_mode == "w":
                writer.writerow(["Index", "Label", "Explanation"])

            for batch in tqdm(dataloader, desc=f"Classifying {relative_path}", leave=False):
                for i in range(len(batch["file_index"])):
                    file_index = batch["file_index"][i]
                    content = batch["content"][i]
                    if not content:
                        continue
                    result = classify_article(chain, content)
                    writer.writerow([file_index, int(result.is_mic), result.explanation])
                f.flush()

        print(f"✅ Finished processing: {folder} -> Output: {csv_file}\n")


if __name__ == "__main__":
    processed_base_dir = Path.cwd().parent / "processed_files"
    classified_base_dir = Path.cwd().parent / "classified_files"
    hf_model_name = "Qwen/Qwen2.5-3B-Instruct"
    process_classification_chain(processed_base_dir, classified_base_dir, hf_model_name)
    print("Classification and saving complete.")

