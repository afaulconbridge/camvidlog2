from typing import Generator, Iterable

import numpy as np
import open_clip
import torch.nn.functional
from PIL import Image


def get_video_embeddings(
    frames: Iterable[tuple[int, np.ndarray]],
) -> Generator[tuple[int, np.ndarray], None, None]:
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
    model, _, processor = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    features = torch.nn.functional.normalize(
        model.encode_text(tokenizer(query)), dim=-1
    )
    return features[0].detach().numpy().astype(np.float32)
