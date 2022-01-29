import numpy as np
from pathlib import Path
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import time
from multi_tile import TileWorker, DeepZoomImageTiler
from multiprocessing import JoinableQueue

def save_patches_SEEGENE(slide_path: str, output_path: str, 
                 resolution_factor: int, tile_size: int,
                 overlap: int, ext: str, use_filter: bool) -> None:
    print(slide_path)
    print('noise filter : ', use_filter)
    workers = 32
    print('multiprocessing : ', workers)
    try:
        queue = JoinableQueue(2*workers)
        slide = open_slide(str(slide_path))
    except Exception as e :
        print(f"Could not open {slide_path}\n")
        print(e)
        return

    W, H = slide.dimensions
    if (W==0) or (H==0):
        print(f"Could not open {slide_path}\n")
        return

    slide_name = Path(slide_path).stem    
    print('slide_name', slide_name)
    for _i in range(workers):
        TileWorker(queue, slide_path, tile_size = tile_size, overlap = overlap, limit_bounds=False, use_filter = use_filter).start()
    tiles = DeepZoomGenerator(
        slide,
        tile_size=tile_size,
        overlap=overlap,
        limit_bounds=False
    )
#     print('end')
    tiler = DeepZoomImageTiler(tiles, Path(output_path), 'jpg', None, queue)
    
    total_time = 0.0
    start = time.time()
    tiler.run()

    for _i in range(workers):
        queue.put(None)
    queue.join()

    total_time += time.time() - start
    print("Titling Time: {:.1f} sec".format(total_time))

    return