from tqdm import tqdm
from pathlib import Path
from utils_tiling import save_patches_SEEGENE
import os

SLIDE_FOLDER = "data/train_data"     # LOCATION OF THE INPUT (MRXS FILES)
TILE_FOLDER = "patches"       # LOCATION FOR THE TILES/PATCHES
TILE_SIZE = 256             # SIZE OF THE TILES/PATCHES
OVERLAP_FACTOR = 0.25       # OVERLAP FACTOR IN PERCENTAGE
RESOLUTION_FACTOR = 4       # RESOLUTION TO CALCULATE ZOOM LEVEL 
EXT = "jpg"                 # FORMAT OF THE TILES. E.G., JPG OR PNG
USE_FILTER = True           # WHETHER FILTER IS ON OR OFF. FALSE = OFF, TRUE = ON


def main(slide_folder: str, tile_folder: str, tile_size: int, resolution_factor: int, overlap_factor: float, ext: str, use_filter: bool) -> None:

    all_slides = list(Path(slide_folder).glob("**/*.mrxs"))
    assert len(all_slides) > 0, f"We did not find slides in {slide_folder}"

    Path(tile_folder).mkdir(exist_ok=True, parents=True)

    overlap = int(tile_size * overlap_factor)
    print(overlap)

    for slide_path in tqdm(all_slides, total=len(all_slides)):
        folder = '/'.join(str(slide_path).split('.')[0].split('/')[1:])
        out_folder = tile_folder + '/' + folder
        save_patches_SEEGENE(
            slide_path=slide_path,
            output_path=out_folder,
            resolution_factor=resolution_factor,
            tile_size=(tile_size - overlap),
            overlap=int(overlap/2),
            ext=ext,
            use_filter=use_filter
        )
    return

if __name__ == "__main__":

    main(
        slide_folder=SLIDE_FOLDER,
        tile_folder=TILE_FOLDER,
        tile_size=TILE_SIZE,
        resolution_factor=RESOLUTION_FACTOR,
        overlap_factor=OVERLAP_FACTOR,
        ext=EXT,
        use_filter=USE_FILTER
    )