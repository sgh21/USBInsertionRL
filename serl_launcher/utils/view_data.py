#! python3

import cv2, sys
import numpy as np
import pickle

def to_display_image(raw_img: np.ndarray) -> np.ndarray:
    """Convert a dataset entry to an OpenCV-friendly uint8 BGR image."""
    img = np.array(raw_img)
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main() -> None:
    # Load the dataset once
    filename = 'dataset/data.pkl'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    cv2.namedWindow("Dataset Sample", cv2.WINDOW_AUTOSIZE)
    print("Viewing dataset samples. Press any key for next image, or 'q'/Esc to quit.")

    try:
        for idx, sample in enumerate(data):
            img_bgr = to_display_image(sample)
            cv2.imshow("Dataset Sample", img_bgr)

            key = cv2.waitKey(0) & 0xFF
            if key in (ord('q'), 27):  # 27 == Esc
                print(f"Stopped at image {idx}.")
                break
    except KeyboardInterrupt:
        print("\nInterrupted with Ctrl-C. Closing viewer...")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
