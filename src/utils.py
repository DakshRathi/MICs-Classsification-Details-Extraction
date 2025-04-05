import torch
from pathlib import Path
from typing import List, Dict, Optional, Union
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI


def get_device() -> str:
    """Determine the available computing device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def init_hf_model(hf_model_name: str) -> HuggingFacePipeline:
    """Initialize a Hugging Face model for text generation."""
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model_kwargs = {"torch_dtype": torch.float16 if device in ["cuda", "mps"] else torch.float32}
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(hf_model_name, **model_kwargs)
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        return_full_text=False
    )
    return HuggingFacePipeline(pipeline=text_generation_pipeline)

def init_llm(hf_model_name: Optional[str] = None) -> Union[HuggingFacePipeline, ChatOpenAI]:
    """Initialize an LLM, using a Hugging Face model if specified, otherwise defaulting to LM Studio."""
    return init_hf_model(hf_model_name) if hf_model_name else ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="LMStudio",
        model_name="qwen2.5-7b-instruct-1m",
        max_tokens=256,
        temperature=0.1
    )

class ArticleDataset(Dataset):
    """Dataset class for loading articles for MIC classification."""

    def __init__(self, article_files: List[Path]): 
        self.article_files = article_files

    def __len__(self) -> int:
        return len(self.article_files)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        article_file = self.article_files[idx]
        content = None # Initialize content to None
        try:
            content = article_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1, ignoring errors
                content = article_file.read_text(encoding="latin-1", errors="ignore")
            except Exception as e:
                # Log or handle the error if even latin-1 fails
                print(f"Warning: Could not read file {article_file} with latin-1: {e}")
                content = None # Ensure content is None if fallback fails
        except FileNotFoundError:
             print(f"Error: File not found during __getitem__: {article_file}")
             content = None
        except Exception as e:
            print(f"Warning: Could not read file {article_file}: {e}")
            content = None # Ensure content is None on other file reading errors

        # Return filename as index, and content
        return {"file_index": article_file.name, "content": content}