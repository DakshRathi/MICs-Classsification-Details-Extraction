# MICs Classification & Details Extraction

## Project Overview
This project focuses on extracting key details from newspaper articles related to militarized interstate conflicts (MICs). The goal is to:

1. Identify whether a MIC occurred.
2. Extract relevant details if MIC occurred.

The dataset consists of newspaper articles retrieved using Boolean search terms related to militarized conflict. However, over 95% of the retrieved articles are false positives, requiring a robust classification system.

## Repository Structure
```
└── 📁MICs-Classsification&Details-Extraction
    ├── 📁data                      # Raw data directory
    ├── 📁results                   # Output directory for all processing stages
    │   ├── 📁processed_files       # Preprocessed article files
    │   ├── 📁classified_files      # Classification results
    │   └── 📁detailed_files        # Extracted details from MIC articles
    ├── 📁src                       # Source code
    │   ├── utils.py                # Shared utilities for LLM and dataset handling
    │   ├── preprocessing.py        # Functions for cleaning and preprocessing text data
    │   ├── classification_chain.py # MIC classification implementation
    │   ├── extraction_chain.py     # Details extraction from MIC articles
    │   ├── unified_pipeline.py     # Combined classification and extraction pipeline
    │   └── main.py                 # CLI entry point for running the pipeline
    ├── .gitignore
    ├── requirements.txt
    └── README.md

```

## Task Description
As part of the preliminary analysis, this project aims to:
- Identify incidents where one country’s forces caused the death of military personnel from another country
- Extract key details:
  - Date of the event
  - Estimated range or exact number of fatalities
  - Countries involved

## Dataset
- **Timeframe:** 2002 - 2023
- **Source:** Retrieved from a curated corpus of newspaper articles
- **File Structure:**
  - `results/classified_files/example_data/2008/classification.csv`: Classification results for MIC identification.
  - `results/classified_files/example_data/2008/details.csv`: Extracted details of MIC events

[Link](https://www.dropbox.com/scl/fo/6dtw8wafbengbze4am7ft/AHUl4WVv-619PJ2YwVFFd1k?rlkey=puwzr74w10ac3lsyom0pfd4y5&e=1&st=gydjujqv&dl=0) to download news articles


## Project Workflow
The implementation follows a structured pipeline. Classification and extraction are implemented using the LangChain framework with options for `Qwen2.5-7B-Instruct-1M` (default llm used with LM Studio) or any Hugging Face model.

1. **Data Preprocessing** (`src/preprocessing.py`):
    - Cleans and preprocesses text data for classification
    - Handles missing values and standardizes formats
    - Preserves directory structure throughout processing

2. **Article Classification** (`src/classification_chain.py`):
    - Uses a few-shot learning approach with carefully crafted examples
    - Outputs binary classification (MIC or non-MIC) with explanations

3. **Details Extraction** (`src/extraction_chain.py`):
    - For articles classified as MICs, extracts:
        - Date of the confrontation
        - Fatality information (minimum and maximum estimates)
        - Countries involved, including initiator and target when possible
    - Results are saved in a structured CSV format

4. **Unified Pipeline** (`src/unified_pipeline.py`):
    - Combines the classification and extraction steps
    - Preserves directory structure from input to output
    - Processes nested directories recursively
    - Implements efficient batch processing for better performance
    - Provides detailed progress tracking with checkpoint support

5. **CLI Tool** (`src/main.py`):
   - Provides a command-line interface for running the unified pipeline
   - Supports preprocessing, classification, and extraction in a single workflow
   - Allows users to specify the Hugging Face model or use LM Studio as the default LLM
   - Automatically handles recursive processing of nested directories while preserving the original directory structure
   - Offers configurable batch size for efficient processing
   - Includes options to skip preprocessing or customize model selection
   - Displays detailed progress tracking with counts of processed articles and identified MICs

6. **ConfliBERT Fine-Tuning** (`notebook/ConfliBERT_Fine_Tuning.ipynb`):
    - Fine-tuned ConfliBERT (a conflict-specific BERT variant) for binary MIC classification
    - Addressed class imbalance using weighted loss functions
    - Leveraged classification results from larger LLMs to create a distilled, specialized model
    - The resulting fine-tuned model requires significantly fewer computational resources while maintaining classification accuracy, enabling faster and more cost-effective processing of large article corpora

## How to Run

### Prerequisites
- Python 3.8+
- Clone this repository and navigate to it:
  ```bash
  git clone https://github.com/DakshRathi/MICs-Classsification-Details-Extraction.git
  ```

  ```bash
  cd MICs-Classsification-Details-Extraction
  ```
- Install required dependencies: 
  ```bash
  pip install -r requirements.txt
  ```

### Example Data
An example file is provided in the `data` folder for testing purposes. You may use this to verify the pipeline works correctly before processing your own data. To use your own data, simply add text files or folders containing text files to the `data` directory. The pipeline will process all text files recursively while preserving the original directory structure.

### Option 1: Using LM Studio (Default)
1. Install **LM Studio** - a free, user-friendly desktop application for running LLMs locally. It provides optimized inference for consumer hardware
2. Download the `Qwen2.5-7B-Instruct-1M` model within LM Studio
3. Start the LM Studio server and ensure it is running
4. Run the unified pipeline:
   ```bash
   python src/main.py ./data ./results
   ```

### Option 2: Using Hugging Face Models
Run the pipeline specifying a Hugging Face model:
```bash
python src/main.py ./data ./results --model Qwen/Qwen2.5-3B-Instruct
```

### Additional Options
```
usage: main.py [-h] [--model MODEL] [--no-preprocess] [--batch-size BATCH_SIZE] raw_data_dir output_dir

Militarized Interstate Confrontation Analysis Pipeline

positional arguments:
  raw_data_dir              Path to directory containing raw text files
  output_dir                Path to directory for storing processed results

options:
  -h, --help                show this help message and exit
  --model MODEL, -m MODEL   Hugging Face model name for classification/extraction
  --no-preprocess           Skip preprocessing step
  --batch-size BATCH_SIZE   Number of articles to process simultaneously (default: 4)
```

## Future Work

Building upon the current progress of classifying and extracting information from around 2,000 articles out of 10,000 from 2008, there are several key areas for future development:

1. **Comprehensive Dataset Processing (2002-2023)**:
    - Process the full range of articles from 2002 to 2023
    - Implement distributed computing for faster processing

2. **Classification Model Enhancement**:
    - Distill knowledge from LLMs into smaller, specialized models
    - Experiment with advanced transformer architectures
    - Implement parameter-efficient fine-tuning techniques

3. **Advanced Entity Recognition**:
    - Develop specialized NER systems for conflict-related entities

4. **Scalability and Efficiency Improvements**:
    - Optimize for cloud deployment
    - Implement progressive model loading for memory efficiency

## Pre-Trained Models
The fine-tuned ConfliBERT model for MIC classification is available in this [Google Drive folder](https://drive.google.com/drive/folders/1bT7poiVpLbPI_pESWDQrtJ2f_GC2o11Q?usp=sharing)