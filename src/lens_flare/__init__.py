import hashlib
import logging
import pickle
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any

import cv2
import imutils
import numpy as np
import skimage
import skimage.exposure
import skimage.measure
from imutils import contours

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


class FileCache:
    def __init__(self, key) -> None:
        self.key = key

    @property
    def _filename(self):
        return hashlib.md5(self.key.encode()).hexdigest()

    def get(self):
        try:
            with open(self._filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    def save(self, obj: Any):
        with open(self._filename, 'wb') as f:
            pickle.dump(obj, f)


@dataclass
class Config:
    src_file: str
    out_file: str
    star_file: str

    threshold_min_percent: float = 99.5
    threshold_max_percent: float = 100.0
    lights_size_min: int = 50
    lights_size_max: int = 500
    star_multiplier: int = 10


class LensFlare:
    def __init__(self, cfg, logger: logging.Logger = None):
        self._cfg = cfg
        self._logger = logger

        self._prepared = False

        self._image = None
        self._blurred = None
        self._contours = None
        self._star = None

    def prepare(self, force=False):
        if self._prepared and not force:
            self._logger.warning('calling prepare() on already prepared LensFlare. skipping.')
            return

        self._logger.info('Preparing LensFlare')

        self._logger.info('Opening star image ...')
        self._star = self._get_star()
        self._logger.info('Star image opened.')
        self._logger.info('Opening source image ...')
        self._image = self._get_image()
        self._logger.info('Source image opened')
        self._logger.info('Calculating blurred image ...')
        self._blurred = self._get_blurred()
        self._logger.info('Blurred calculated.')
        self._logger.info('Calculating contours ...')
        self._contours = self._get_contours()
        self._logger.info('Contours calculated.')
        self._prepared = True

    def _get_image(self):
        return cv2.imread(self._cfg.src_file)

    def _get_star(self):
        return cv2.imread(self._cfg.star_file, -1)

    def _get_blurred(self):
        gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (11, 11), 0)

    def _get_contours(self):
        _cache = FileCache(f'{self._cfg.src_file}_get_contours')
        _cached_value = _cache.get()
        if _cached_value:
            return _cached_value

        min_thresh_value = np.percentile(
            self._blurred, self._cfg.threshold_min_percent)
        max_thresh_value = np.percentile(
            self._blurred, self._cfg.threshold_max_percent)

        actions = [
            lambda img: cv2.threshold(
                img, min_thresh_value, max_thresh_value, cv2.THRESH_TOZERO)[1],
            lambda img: cv2.erode(img, None, iterations=4),
            lambda img: cv2.dilate(img, None, iterations=4)
        ]

        image = self._blurred.copy()

        for a in actions:
            image = a(image)

        # https://en.wikipedia.org/wiki/Connected-component_labeling
        labels = skimage.measure.label(image, connectivity=2, background=0)
        mask = np.zeros(image.shape, dtype="uint8")

        for label in np.unique(labels):
            if label == 0:
                continue

            # create mask for the label
            labelMask = np.zeros(image.shape, dtype="uint8")
            labelMask[labels == label] = 255

            # count pixels in this mask
            numPixels = cv2.countNonZero(labelMask)

            # filter label by number of pixels
            if self._cfg.lights_size_max > numPixels > self._cfg.lights_size_min:
                mask = cv2.add(mask, labelMask)

        cnts = cv2.findContours(image, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = contours.sort_contours(cnts)[0]

        _cache.save(cnts)

        return cnts

    @staticmethod
    def _get_offset_point(i, x, y):
        h, w, _ = i.shape
        x_offset = int(w / 2)
        y_offset = int(h / 2)
        return int(x - x_offset), int(y - y_offset)

    @staticmethod
    def _overlay_image_alpha(img, img_overlay, pos, alpha_mask):
        """Overlay img_overlay on top of img at the position specified by
        pos and blend using alpha_mask.
        Alpha mask must contain values within the range [0, 1] and be the
        same size as img_overlay.
        """

        x, y = pos

        # Image ranges
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

        # Overlay ranges
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        channels = img.shape[2]

        alpha = alpha_mask[y1o:y2o, x1o:x2o]
        alpha_inv = 1.0 - alpha

        for c in range(channels):
            img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                    alpha_inv * img[y1:y2, x1:x2, c])

    def place_star(self, c, image):
        # draw the bright spot on the image
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        # get rectangle of source image and match colors to sample
        (x, y, w, h) = cv2.boundingRect(c)
        y1 = int((y + (h / 2)) - (h / 4))
        y2 = int((y + (h / 2)) + (h / 4))
        x1 = int((x + (w / 2)) - (w / 4))
        x2 = int((x + (w / 2)) + (w / 4))
        crop_img = image[y1:y2, x1:x2]
        matched = skimage.exposure.match_histograms(self._star, crop_img, multichannel=False)
        new_size = (
            int(radius * self._cfg.star_multiplier),
            int(radius * self._cfg.star_multiplier)
        )
        resized = cv2.resize(matched[:, :, 0:3], new_size, interpolation=cv2.INTER_AREA)
        alpha_channel = self._star[:, :, 3]
        resized_alpha = cv2.resize(alpha_channel, new_size, interpolation=cv2.INTER_AREA)
        self._overlay_image_alpha(image, resized, self._get_offset_point(resized, cX, cY), resized_alpha / 255.0)

        return image

    def place_stars_on_image(self, iterator: bool = False):
        # place stars on image

        image = self._image.copy()

        prev = image
        for (i, c) in enumerate(self._contours):
            image = self.place_star(c, image.copy())

            if iterator:
                skip = yield image
                if skip:
                    logger.info(f'skipping step {i}')
                    image = prev

            prev = image

        yield image


class AutoLensFlare(LensFlare):
    def process(self):
        self._logger.info(f'Config is {self._cfg}')
        self.prepare()
        self._logger.info('Placing stars on image ...')
        out_image = next(self.place_stars_on_image())
        self._logger.info('Stars has been placed on image.')
        self._logger.info('Saving output image')
        cv2.imwrite(self._cfg.out_file, out_image)
        self._logger.info(f'Output image saved: {self._cfg.out_file}')


class StepByStepLensFlare(LensFlare):
    def process(self):
        self._logger.info(f'Config is {self._cfg}')
        self.prepare()
        self._logger.info('Placing stars on image step by step...')

        out_image = None

        iterator = self.place_stars_on_image(iterator=True)
        skip = None
        i = 0  # just for logging
        while True:
            try:
                out_image = iterator.send(skip)
            except StopIteration:
                break
            else:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_file_name = tmp_dir + uuid.uuid4().hex + '.jpg'

                    with open(tmp_file_name, 'wb') as new_file:
                        new_file.write(out_image)

                    logger.info(f'saving {tmp_file_name}')

                    cv2.imwrite(tmp_file_name, out_image)

                    skip = yield tmp_file_name

                    logger.info(f'step {i}: skip={skip} for process image')

            i += 1

        if out_image is None:
            logger.warning(f'Can\'t save image')
            return

        self._logger.info('Stars has been placed on image.')
        self._logger.info('Saving output image')
        cv2.imwrite(self._cfg.out_file, out_image)
        self._logger.info(f'Output image saved: {self._cfg.out_file}')
