import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Optional, Dict
import logging
from pathlib import Path

# Configure logging for server deployment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextSummarizer:
    """A class to perform text summarization using a single transformer model.
    Truncates input to 1024 tokens and processes it directly.
    """

    # Class constants
    MAX_INPUT_LENGTH: int = 1024
    MAX_SUMMARY_LENGTH: int = 512

    def __init__(
        self,
        model_path: str = '../best_model',
        tokenizer_path: str = '../best_model'
    ):
        """Initialize the TextSummarizer with model and tokenizer paths.

        Args:
            model_path (str): Path to the summarization model.
            tokenizer_path (str): Path to the tokenizer.
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self._device}")
        self._model_path = Path(model_path)
        self._tokenizer_path = Path(tokenizer_path)
        self._model: Optional[AutoModelForSeq2SeqLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the summarization model and tokenizer."""
        try:
            logger.info(f"Loading model from {self._model_path}")
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_path).to(self._device)
            logger.info(f"Loading tokenizer from {self._tokenizer_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def summarize(self, text: str) -> str:
        """Summarize the input text by truncating to 1024 tokens and processing directly.

        Args:
            text (str): Input text to summarize.

        Returns:
            str: Final summarized text or error message if processing fails.
        """
        if not text or not isinstance(text, str):
            logger.error("Invalid input: text must be a non-empty string")
            return "Invalid input: text must be a non-empty string."

        try:
            # Step 1: Tokenize and truncate input text to MAX_INPUT_LENGTH
            inputs = self._tokenizer(
                text,
                max_length=self.MAX_INPUT_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                return_token_type_ids=False
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Step 2: Generate summary
            self._model.eval()
            with torch.no_grad():
                summary_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.MAX_SUMMARY_LENGTH,
                    num_beams=4,
                    length_penalty=1.0
                )

                # Step 3: Decode summary
                final_summary = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
                if final_summary and not final_summary.endswith("."):
                    final_summary += "."

            logger.info("Summarization completed successfully")
            return final_summary

        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return f"Error during summarization: {str(e)}"