import cv2
import os

from comparison.character_analysis.char_segmenter import (
    extract_character_boxes,
    draw_character_boxes,
)


def debug_character_segmentation(image_path, output_dir="char_debug"):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    boxes, th = extract_character_boxes(img)

    vis = draw_character_boxes(img, boxes)

    cv2.imwrite(
        os.path.join(output_dir, "characters_detected.jpg"),
        vis,
    )

    if th is not None:
        cv2.imwrite(
            os.path.join(output_dir, "thresholded.jpg"),
            th,
        )

    print(f"Characters detected: {len(boxes)}")

    for i, box in enumerate(boxes, 1):
        print(f"{i}: {box}")

    print(
        f"Saved results to: {output_dir}"
    )