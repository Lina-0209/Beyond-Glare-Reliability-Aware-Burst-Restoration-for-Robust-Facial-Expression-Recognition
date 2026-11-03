import unittest
import numpy as np

from vision_module.aligned_mean import aligned_mean_rgb


class AlignedMeanTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[4, 4], [24, 4], [4, 24], [24, 24]], dtype=np.float32)
        self.landmarks = np.repeat(self.points[None], 4, axis=0)

    def test_identical_frames_preserved(self):
        image = np.random.RandomState(42).randint(0, 256, (32, 32, 3)).astype(np.uint8)
        output = aligned_mean_rgb([image] * 4, self.landmarks, image_size=32)
        np.testing.assert_array_equal(output, image)

    def test_equal_weights(self):
        images = [np.full((32, 32, 3), value, dtype=np.uint8) for value in [0, 40, 80, 120]]
        output = aligned_mean_rgb(images, self.landmarks, image_size=32)
        np.testing.assert_array_equal(output, np.full_like(images[0], 60))

    def test_translation_is_undone(self):
        anchor = np.zeros((32, 32, 3), dtype=np.uint8)
        anchor[10:20, 10:20] = 200
        offsets = [0, 2, -2, 1]
        frames = [np.roll(anchor, offset, axis=1) for offset in offsets]
        landmarks = self.landmarks.copy()
        for i, offset in enumerate(offsets):
            landmarks[i, :, 0] += offset
        output = aligned_mean_rgb(frames, landmarks, image_size=32)
        np.testing.assert_array_equal(output[4:-4, 4:-4], anchor[4:-4, 4:-4])

    def test_reject_malformed(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            aligned_mean_rgb([image] * 3, self.landmarks)
        with self.assertRaises(ValueError):
            aligned_mean_rgb([image] * 4, self.points)
        bad = self.landmarks.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            aligned_mean_rgb([image] * 4, bad)


if __name__ == "__main__":
    unittest.main()
