from dataclasses import dataclass


@dataclass
class Images:
    grayscale_filenames: str


@dataclass
class Config:
    images: Images
