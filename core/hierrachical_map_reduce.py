from typing import List, Dict, Any, Optional
from langchain.schema import Document
import numpy as np
from tqdm import tqdm
from pre_trained_model.phi35mini import GenerationConfig, Phi35MiniSummarizer
import logging


class HierarchicalMapReduceSummarizer:
    """
    Implements a sophisticated hierarchical map-reduce approach for document summarization.
    This approach processes documents in multiple stages with adaptive chunking and
    targeted summarization strategies.
    """

    # Different prompt templates for different stages of summarization
    MAP_PROMPT = """Summarize the following content with focus on preserving key information, 
facts, and main points. Be comprehensive and maintain all important details:

{text}

Summary:"""

    REDUCE_PROMPT = """Synthesize the following summaries into a coherent, comprehensive summary.
Focus on maintaining key themes, crucial details, and logical flow across all input summaries:

{text}

Synthesized summary:"""

    FINAL_PROMPT = """Create a polished, comprehensive summary of the following content.
Ensure you capture main themes, key details, essential facts, and logical connections.
The summary should be well-structured and read as a cohesive whole:

{text}

Final summary:"""

    def __init__(
        self,
        summarizer: Optional[Phi35MiniSummarizer] = None,
        max_input_size: int = 16000,
        max_map_chunk_size: int = 8000,
        min_chunk_size: int = 2000,
        map_tokens: int = 768,
        reduce_tokens: int = 1024,
        final_tokens: int = 1536,
        overlap_percentage: float = 0.1,
    ):
        """
        Initialize the hierarchical map-reduce summarizer.

        Args:
            summarizer: The Phi-3.5 summarizer to use (will create one if not provided)
            max_input_size: Maximum cumulative tokens to process in a single reduce step
            max_map_chunk_size: Maximum size of individual chunks in the map phase
            min_chunk_size: Minimum size of text chunks to process
            map_tokens: Maximum tokens to generate in map phase
            reduce_tokens: Maximum tokens to generate in reduce phase
            final_tokens: Maximum tokens to generate in final summary
            overlap_percentage: Percentage of overlap between chunks
        """
        self.summarizer = summarizer or Phi35MiniSummarizer()
        self.max_input_size = max_input_size
        self.max_map_chunk_size = max_map_chunk_size
        self.min_chunk_size = min_chunk_size
        self.map_tokens = map_tokens
        self.reduce_tokens = reduce_tokens
        self.final_tokens = final_tokens
        self.overlap_percentage = overlap_percentage
        self.logger = logging.getLogger(__name__)

        # Set up configuration templates
        self.map_config = GenerationConfig(
            max_new_tokens=map_tokens,
            min_new_tokens=int(map_tokens * 0.3),
            temperature=0.2,
            repetition_penalty=1.2,
        )

        self.reduce_config = GenerationConfig(
            max_new_tokens=reduce_tokens,
            min_new_tokens=int(reduce_tokens * 0.3),
            temperature=0.15,
            repetition_penalty=1.3,
        )

        self.final_config = GenerationConfig(
            max_new_tokens=final_tokens,
            min_new_tokens=int(final_tokens * 0.3),
            temperature=0.1,
            repetition_penalty=1.4,
            num_beams=5,  # More beams for final summary
        )

    def _adaptive_chunk_documents(self, documents: List[Document]):
        """
        Adaptively chunk documents based on their content and structure.

        Args:
            documents: List of document objects

        Returns:
            List of text chunks with appropriate size and overlap
        """
        # Start with a simple approach and extract text from documents
        texts = [doc.page_content for doc in documents]
        combined_text = "\n\n".join(texts)

        # If total text is small enough, don't chunk further
        if len(combined_text) < self.min_chunk_size:
            return [combined_text]

        # Otherwise, split into appropriately sized chunks with overlap
        chunks = []
        total_chars = len(combined_text)

        # Estimate optimal chunk size based on total content
        # More content = larger chunks to minimize hierarchy depth
        if total_chars > 100000:
            chunk_size = min(self.max_map_chunk_size, 10000)
        elif total_chars > 50000:
            chunk_size = min(self.max_map_chunk_size, 8000)
        else:
            chunk_size = min(self.max_map_chunk_size, 5000)

        overlap = int(chunk_size * self.overlap_percentage)

        # Create chunks with overlap
        position = 0
        while position < total_chars:
            chunk_end = min(position + chunk_size, total_chars)

            # Try to find a natural breakpoint (paragraph or sentence)
            if chunk_end < total_chars:
                # Look for paragraph break
                paragraph_break = combined_text.rfind("\n\n", position, chunk_end)
                if (
                    paragraph_break != -1
                    and paragraph_break > position + chunk_size * 0.7
                ):
                    chunk_end = paragraph_break + 2
                else:
                    # Look for sentence break (period followed by space)
                    sentence_break = combined_text.rfind(". ", position, chunk_end)
                    if (
                        sentence_break != -1
                        and sentence_break > position + chunk_size * 0.8
                    ):
                        chunk_end = sentence_break + 2

            chunks.append(combined_text[position:chunk_end])
            position = chunk_end - overlap

        return chunks

    def _map_phase(self, chunks: List[str]):
        """
        Execute the map phase by summarizing each chunk independently.

        Args:
            chunks: List of text chunks to summarize

        Returns:
            List of summaries for each chunk
        """
        self.logger.info(f"Map phase: Processing {len(chunks)} chunks")
        summaries = []

        for i, chunk in enumerate(tqdm(chunks, desc="Map Phase")):
            try:
                # Create a custom prompt for this chunk
                custom_prompt = self.MAP_PROMPT.format(text=chunk)

                # Generate summary
                summary = self.summarizer.generate_summary(
                    text=chunk,
                    config=self.map_config,
                    custom_config={"prompt_template": custom_prompt},
                )

                summaries.append(summary)
                self.logger.debug(
                    f"Chunk {i+1}/{len(chunks)}: Summarized to {len(summary.split())} words"
                )

            except Exception as e:
                self.logger.error(f"Error summarizing chunk {i+1}: {str(e)}")
                # Add abbreviated chunk as fallback
                summaries.append(chunk[:500] + "...")

        return summaries

    def _reduce_phase(self, summaries: List[str], level: int = 1):
        """
        Execute the reduce phase by combining summaries hierarchically.

        Args:
            summaries: List of summaries to combine
            level: Current level in the reduction hierarchy

        Returns:
            Single combined summary
        """
        if not summaries:
            return ""

        # If we only have one summary or they're small enough, combine directly
        combined_text = "\n\n".join(summaries)
        if len(summaries) == 1 or len(combined_text) < self.max_input_size:
            self.logger.info(
                f"Reduce level {level}: Direct combination of {len(summaries)} summaries"
            )

            # Choose appropriate prompt and config based on level
            if level == 1 and len(summaries) > 5:
                prompt = self.REDUCE_PROMPT.format(text=combined_text)
                config = self.reduce_config
            else:
                prompt = self.FINAL_PROMPT.format(text=combined_text)
                config = self.final_config

            # Generate combined summary
            return self.summarizer.generate_summary(
                text=combined_text,
                config=config,
                custom_config={"prompt_template": prompt},
            )

        # For larger sets, combine recursively in balanced groups
        self.logger.info(
            f"Reduce level {level}: Recursive combination of {len(summaries)} summaries"
        )

        # Group summaries into batches
        batches = []
        current_batch = []
        current_length = 0

        for summary in summaries:
            summary_length = len(summary)

            # If adding this summary would exceed max size, start a new batch
            if current_batch and current_length + summary_length > self.max_input_size:
                batches.append(current_batch)
                current_batch = [summary]
                current_length = summary_length
            else:
                current_batch.append(summary)
                current_length += summary_length

        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)

        # Process each batch and collect results
        batch_summaries = []
        for i, batch in enumerate(batches):
            self.logger.debug(
                f"Processing batch {i+1}/{len(batches)} with {len(batch)} summaries"
            )
            batch_text = "\n\n".join(batch)

            # Use reduce prompt for intermediate batches
            prompt = self.REDUCE_PROMPT.format(text=batch_text)
            batch_summary = self.summarizer.generate_summary(
                text=batch_text,
                config=self.reduce_config,
                custom_config={"prompt_template": prompt},
            )
            batch_summaries.append(batch_summary)

        # Recursively combine the batch summaries
        return self._reduce_phase(batch_summaries, level + 1)

    def summarize(self, documents: List[Document]):
        """
        Summarize a collection of documents using hierarchical map-reduce.

        Args:
            documents: List of documents to summarize

        Returns:
            Dictionary containing the final summary and metadata
        """
        self.logger.info(f"Starting summarization of {len(documents)} documents")

        # Phase 1: Adaptive chunking
        chunks = self._adaptive_chunk_documents(documents)
        self.logger.info(f"Chunked into {len(chunks)} segments")

        # Phase 2: Map (summarize each chunk)
        chunk_summaries = self._map_phase(chunks)

        # Phase 3: Reduce (combine summaries hierarchically)
        final_summary = self._reduce_phase(chunk_summaries)

        # Track metadata for analysis
        word_count = len(final_summary.split())
        compression_ratio = sum(
            len(doc.page_content.split()) for doc in documents
        ) / max(1, word_count)

        return {
            "summary": final_summary,
            "metadata": {
                "original_documents": len(documents),
                "chunks_processed": len(chunks),
                "word_count": word_count,
                "compression_ratio": compression_ratio,
            },
        }
