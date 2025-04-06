import cv2
import numpy as np

class Plotter:

    def __init__(self):
        cv2.namedWindow("pose")

    def plot(self, img: np.ndarray, points: list[tuple[float, ...]]) -> np.ndarray:

        for i, (x, y, _) in enumerate(points[0]):
            x = int(x)
            y = int(y)
            if 0 < x < img.shape[1] and 0 < y < img.shape[0]:
                cv2.circle(img, [int(x), int(y)], 4, (0, 255, 0), -1)
        return img
    
    def show(self, img, points: list[tuple[float, ...]] | None = None) -> int:
        if points is not None:
            img = self.plot(img, points)
        
        cv2.imshow("pose", img)
        return int(cv2.waitKey(1))