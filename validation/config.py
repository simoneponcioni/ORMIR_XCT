from dataclasses import dataclass


@dataclass
class Images:
    grayscale_filenames: str
    folder_id: str
    

@dataclass
class Config:
    images: Images
