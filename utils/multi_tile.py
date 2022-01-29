import os
import numpy as np
from skimage.measure import block_reduce
from multiprocessing import Process
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator

def pooling_img(img: np.ndarray, scale_size: int = 16) -> np.ndarray:
    block_size = (img.shape[0] // scale_size, img.shape[1] // scale_size, 1)
    return block_reduce(image=img, block_size=block_size, func=np.average)


def count_purple_box(pooled: np.ndarray) -> int:
    r, g, b = pooled[..., 0], pooled[..., 1], pooled[..., 2]
    c1 = r > g - 10
    c2 = b > g - 10
    c3 = (r + b) / 2 > g + 20
    return pooled[c1 & c2 & c3].size

def count_black_box(pooled: np.ndarray) -> int:
    r, g, b = pooled[..., 0], pooled[..., 1], pooled[..., 2]
    c1 = r < 10
    c2 = g < 10
    c3 = b < 10
    return pooled[c1 & c2 & c3].size

def noise_patch(img: np.ndarray, purple_thershold: float = 0.3, black_thershold: float = 0.1) -> bool:

    pooled = pooling_img(img)
    purple_n = count_purple_box(pooled)
    black_n = count_black_box(pooled)
    total = pooled.size    
    c1 = black_n / total < black_thershold
    c2 = purple_n / total >= purple_thershold
    return not c1 & c2
    
class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds, use_filter,
                quality=None):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._user_filter = use_filter
        self._quality = None
        self._slide = None
#         print('TileWorker is initialized')

    def run(self):
        self._slide = open_slide(str(self._slidepath))
        last_associated = None
        dz = self._get_dz()
#         print('TileWorker is called')
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            associated, level, address, outfile = data
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated
            tile = dz.get_tile(level, address)
            
            if not self._user_filter:
                tile.save(outfile)
            elif not noise_patch(np.array(tile)):
                tile.save(outfile)

            self._queue.task_done()

    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)


class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz, basename, format, associated, queue):
        self._dz = dz
        self._basename = basename
        self._format = format
        self._associated = associated
        self._queue = queue
        self._processed = 0

    def run(self):
        self._write_tiles()
        self._write_dzi()

    def _write_tiles(self):
        level = 16
        tiledir=self._basename
        if not os.path.exists(tiledir):
            os.makedirs(tiledir)
        cols, rows = self._dz.level_tiles[level]
        for row in range(1,rows-1):
            for col in range(1,cols-1):
                tilename = os.path.join(tiledir, '%d_%d.%s' % (
                                row, col, self._format))
                if not os.path.exists(tilename):
                    self._queue.put((self._associated, level, (col, row),
                                tilename))
                self._tile_done()

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz.tile_count

    def _write_dzi(self):
        with open('%s.dzi' % self._basename, 'w') as fh:
            fh.write(self.get_dzi())

    def get_dzi(self):
        return self._dz.get_dzi(self._format)

