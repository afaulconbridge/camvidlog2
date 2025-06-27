from typing import Generator, Iterable

import numpy as np
import open_clip
import torch.nn.functional
from PIL import Image


def get_video_embeddings(
    frames: Iterable[tuple[int, np.ndarray]],
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Generate embeddings for video frames using BioClip model.

    Args:
        frames: An iterable of tuples containing (frame_number, frame_array),
            where frame_array is a numpy array representing a video frame.

    Yields:
        tuple[int, np.ndarray]: A tuple containing (frame_number, embedding),
            where embedding is a 512-dimensional normalized float32 vector.

    Notes:
        - The embeddings are L2-normalized, meaning their magnitude will be close to 1.0
        - This uses the BioClip model from https://huggingface.co/imageomics/bioclip
        - Processing frames in batches could be more efficient than processing one at a time
    """
    model, _, processor = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip"
    )
    model.eval()
    for frame_no, frame in frames:
        frame_pil = Image.fromarray(frame.squeeze())
        image_features = torch.nn.functional.normalize(
            model.encode_image(processor(frame_pil).unsqueeze(0)),
            dim=-1,
        )

        yield (
            frame_no,
            image_features[0].detach().numpy().astype(np.float32),
        )


def get_string_embedding(query: str) -> np.ndarray:
    """Get a 512-dimensional embedding for a single text query using BioClip model.

    Args:
        query: The text string to be embedded

    Returns:
        np.ndarray: A 1D array of shape (512,) containing the L2-normalized embedding vector
                  with magnitude close to 1.0

    Notes:
        - This uses the BioClip model from https://huggingface.co/imageomics/bioclip
        - Each embedding is normalized using L2 normalization
    """
    return get_string_embeddings([query])[0]


def get_string_embeddings(queries: Iterable[str]) -> np.ndarray:
    """Get 512-dimensional embeddings for multiple text queries using BioClip model.

    Args:
        queries: An iterable of text strings to be embedded

    Returns:
        np.ndarray: A 2D array of shape (n, 512) where each row contains the L2-normalized
            embedding vector for a query, with magnitude close to 1.0

    Notes:
        - This uses the BioClip model from https://huggingface.co/imageomics/bioclip
        - The embeddings are generated in batches for efficiency
        - Each embedding is normalized using L2 normalization
    """
    model, _, processor = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    features = model.encode_text(tokenizer(list(queries)), normalize=True)
    # an n x 512 matrix, where each row corresponds to a query string
    return features.detach().numpy().astype(np.float32)
