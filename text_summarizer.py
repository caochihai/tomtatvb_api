import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Optional, Dict
import logging
import re
from pathlib import Path

# Configure logging for server deployment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextSummarizer:
    """A class to perform text summarization using a single transformer model.
    Supports intelligent chunking with batch processing for long texts and GPU/CPU compatibility.
    """

    # Class constants
    MAX_INPUT_LENGTH: int = 1024
    MAX_SUMMARY_LENGTH: int = 512
    CHUNK_BATCH_SIZE_MAP: Dict[int, int] = {2: 2, 8: 4, float('inf'): 8}  # Chunk count to batch size mapping

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

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks based on sentence boundaries and token length.

        Args:
            text (str): Input text to be chunked.

        Returns:
            List[str]: List of text chunks within token limits.
        """
        try:
            sentences = re.split(r'(?<=[\.\n])\s+', text)  # Split by period or newline
            chunks, current_chunk = [], ""
            for sentence in sentences:
                if not sentence.strip():
                    continue
                temp_chunk = current_chunk + " " + sentence if current_chunk else sentence
                tokenized_len = len(self._tokenizer.encode(temp_chunk, truncation=False))
                if tokenized_len <= self.MAX_INPUT_LENGTH:
                    current_chunk = temp_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
            if current_chunk:
                chunks.append(current_chunk.strip())
            return chunks
        except Exception as e:
            logger.error(f"Failed to split text into chunks: {str(e)}")
            return []

    def summarize(self, text: str) -> str:
        """Summarize the input text by chunking and batch processing, combining summaries with periods.

        Args:
            text (str): Input text to summarize.

        Returns:
            str: Final summarized text or error message if processing fails.
        """
        if not text or not isinstance(text, str):
            logger.error("Invalid input: text must be a non-empty string")
            return "Invalid input: text must be a non-empty string."

        try:
            # Step 1: Split text into chunks
            chunks = self._split_text_into_chunks(text)
            if not chunks:
                logger.warning("No valid chunks created from input text")
                return "Không thể phân đoạn văn bản để tóm tắt."

            # Step 2: Summarize chunks in batches
            chunk_summaries = []
            chunk_batch_size = next(size for threshold, size in self.CHUNK_BATCH_SIZE_MAP.items() if len(chunks) <= threshold)
            self._model.eval()
            with torch.no_grad():
                for i in range(0, len(chunks), chunk_batch_size):
                    chunk_batch = chunks[i:i + chunk_batch_size]
                    try:
                        # Tokenize all chunks in the batch
                        inputs = self._tokenizer(
                            chunk_batch,
                            max_length=self.MAX_INPUT_LENGTH,
                            truncation=True,
                            padding="max_length",
                            return_tensors="pt",
                            return_token_type_ids=False
                        )
                        inputs = {k: v.to(self._device) for k, v in inputs.items()}
                        # Generate summaries for the batch
                        summary_ids = self._model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_length=self.MAX_SUMMARY_LENGTH,
                            num_beams=4,
                            length_penalty=1.0
                        )
                        # Decode summaries
                        batch_summaries = [
                            self._tokenizer.decode(summary_id, skip_special_tokens=True).strip()
                            for summary_id in summary_ids
                        ]
                        chunk_summaries.extend(batch_summaries)
                    except Exception as e:
                        logger.error(f"Error generating summaries for chunk batch {i//chunk_batch_size + 1}: {str(e)}")
                        chunk_summaries.extend([""] * len(chunk_batch))

            # Step 3: Combine summaries with periods
            final_summary = ". ".join(chunk_summaries).strip()
            if final_summary and not final_summary.endswith("."):
                final_summary += "."

            logger.info("Summarization completed successfully")
            return final_summary

        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return f"Error during summarization: {str(e)}"
