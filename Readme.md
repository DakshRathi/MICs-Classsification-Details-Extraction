# MICs Classification & Details Extraction

## Project Overview
This project focuses on extracting key details from newspaper articles related to militarized interstate conflicts (MICs). The goal is to:

1. Identify whether a MIC occurred.
2. Extract relevant details if MIC occurred.

The dataset consists of newspaper articles retrieved using Boolean search terms related to militarized conflict. However, over 95% of the retrieved articles are false positives, requiring a robust classification system.

## Repository Structure
```
└── 📁MICs-Classsification&Details-Extraction
    └── 📁classified_files         # CSV files with classification results
        └── 2008_classification.csv
    └── 📁detailed_files           # CSV files with extracted details from MIC articles
        └── 2008_mic_details.csv
    └── 📁notebook                 # Implementation notebooks
        └── 1_data_preprocessing.ipynb    # Preprocessing raw articles
        └── 2_article_classifier.ipynb    # MIC classification implementation
        └── 3_details_extraction.ipynb    # Details extraction from MIC articles
        └── 4_unified_pipeline.ipynb      # Combined classification and extraction
        └── 5_ConfliBERT_Fine_Tuning.ipynb    # Fine-tuning ConfliBERT for classification
    └── 📁processed_files          # Preprocessed article files
    └── 📁raw_data                 # Original article corpus
    └── .gitignore
    └── requirements.txt
    └── Readme.md
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
  - `classified_files/2008_classification.csv`: Classification results for MIC identification.\
  - `detailed_files/2008_mic_details.csv`: Extracted details of MIC events
  - `notebook/`: Contains Jupyter notebooks for different stages of the pipeline

[Link](https://www.dropbox.com/scl/fo/6dtw8wafbengbze4am7ft/AHUl4WVv-619PJ2YwVFFd1k?rlkey=puwzr74w10ac3lsyom0pfd4y5&e=1&st=gydjujqv&dl=0) to download news articles


## Project Workflow
The implementation follows a structured pipeline. Classification and extraction are implemented using the LangChain framework with `Qwen2.5-7B-Instruct-1M` as the base LLM.

1. **Data Preprocessing** (`notebook/1_data_preprocessing.ipynb`):
   - Cleans and preprocesses text data for classification
   - Handles missing values and standardizes formats

2. **Article Classification** (`notebook/2_article_classifier.ipynb`):
   - Uses a few-shot learning approach with carefully crafted examples
   - Outputs binary classification (MIC or non-MIC) with explanations

3. **Details Extraction** (`notebook/3_details_extraction.ipynb`):
    - For articles classified as MICs, extracts:
        - Date of the confrontation
        - Fatality information (minimum and maximum estimates)
        - Countries involved, including initiator and target when possible
    - Results are saved in a structured CSV format

4. **Unified Pipeline** (`notebook/4_unified_pipeline.ipynb`):
   - Combines the classification and extraction steps
   - Creates a seamless workflow from processed data to structured output

5. **ConfliBERT Fine-Tuning** (`notebook/5_ConfliBERT_Fine_Tuning.ipynb`):
    - Fine-tuned ConfliBERT (a conflict-specific BERT variant) for binary MIC classification
    - Addressed class imbalance using weighted loss functions.
    - Leveraged the classification results from the Qwen2.5-7B-Instruct-1M LLM to create a distilled, specialized model.
    - This approach transfers knowledge from the large 7B parameter model to a much smaller, computationally efficient model for future classification.
    - The resulting fine-tuned model requires significantly fewer computational resources while maintaining classification accuracy, enabling faster and more cost-effective processing of large article corpora.

## Future Work
Building upon the current progress of classifying and extracting information from 2,081 articles out of 10,218 from 2008, there are several key areas for future development and expansion of this project:

1. Comprehensive Dataset Processing (2002-2023):
    - Extend the existing pipeline to process the full range of articles from 2002 to 2023
    - Implement efficient batch processing techniques to handle the increased volume of data

2. Classification Model Enhancement:
    - Distill knowledge from LLMs into smaller, specialized models by using the LLM-generated classifications as training data, creating a more efficient classification pipeline.
    - Experiment with advanced transformer architectures to potentially improve classification accuracy and reduce inference time.

3. Advanced Entity Recognition:
    - Develop a more sophisticated named entity recognition (NER) system specifically tailored for extracting fatality information and countries involved.
    - Implement a custom NER model trained on a dataset of conflict-related texts to improve accuracy in identifying relevant entities.

4. Scalability and Efficiency Improvements:
    - Optimize the current pipeline for processing speed and memory efficiency.
## How to Run
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install **LM Studio** and download the `Qwen2.5-7B-Instruct-1M` model.
4. Start the LM Studio server and ensure it is running.
5. Run the notebooks sequentially for classification and details extraction.

## Pre-Trained Models
The fine-tuned ConfliBERT model for MIC classification is available in this [Google Drive folder](https://drive.google.com/drive/folders/1bT7poiVpLbPI_pESWDQrtJ2f_GC2o11Q?usp=sharing)
