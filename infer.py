import argparse

import cv2
import numpy as np
import onnxruntime as ort

from depth_anything.util.transform import load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="device number",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to ONNX model.",
    )
    return parser.parse_args()


def infer(device: int, model: str, viz: bool = True):
    session = ort.InferenceSession(
        model, providers=[
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
    )
    i_n, i_c, i_h, i_w = session.get_inputs()[0].shape
    cap = cv2.VideoCapture(device)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(
        filename='output.mp4',
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(w, h*2),
    )

    while True:
        orig_image, inference_image, (orig_h, orig_w) = load_image(cap, i_h, i_w)
        if orig_image is None:
            break

        depth = session.run(None, {"input": inference_image})[0]

        depth = cv2.resize(depth[0, 0], (orig_w, orig_h))
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        # Visualization
        if viz:
            margin_width = 50
            caption_height = 60
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            split_region = np.ones((orig_h, margin_width, 3), dtype=np.uint8) * 255
            combined_results = cv2.hconcat([orig_image, split_region, depth_color])

            caption_space = (
                np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8)
                * 255
            )
            captions = ["Raw image", "Depth Anything"]
            segment_width = orig_w + margin_width
            for i, caption in enumerate(captions):
                # Calculate text size
                text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

                # Calculate x-coordinate to center the text
                text_x = int((segment_width * i) + (orig_w - text_size[0]) / 2)

                # Add text caption
                cv2.putText(
                    caption_space,
                    caption,
                    (text_x, 40),
                    font,
                    font_scale,
                    (0, 0, 0),
                    font_thickness,
                )

            final_result = cv2.vconcat([caption_space, combined_results])

            cv2.imshow("depth", final_result)
            video_writer.write(np.vstack([orig_image, depth_color]))
            key = cv2.waitKey(1)
            if key == 27: # ESC
                break

    if video_writer:
        video_writer.release()

    if cap:
        cap.release()


if __name__ == "__main__":
    args = parse_args()
    infer(**vars(args))
