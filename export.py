import argparse

import torch
from onnx import load_model, save_model
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

from depth_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["s", "b", "l"],
        required=True,
        help="Model size variant. Available options: 's', 'b', 'l'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        required=False,
        help="Path to save the ONNX model.",
    )

    return parser.parse_args()


def export_onnx(model: str, output: str = None):
    # Handle args
    if output is None:
        output = f"depth_anything_vit{model}14.onnx"

    # Device for tracing (use whichever has enough free memory)
    device = "cpu"

    # Sample image for tracing (dimensions don't matter)
    # image, _ = load_image("assets/sacre_coeur1.jpg")
    # image = torch.from_numpy(image).to(device)

    # Load model params
    if model == "s":
        depth_anything = DPT_DINOv2(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384]
        )
    elif model == "b":
        depth_anything = DPT_DINOv2(
            encoder="vitb", features=128, out_channels=[96, 192, 384, 768]
        )
    else:  # model == "l"
        depth_anything = DPT_DINOv2(
            encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]
        )

    depth_anything.to(device).load_state_dict(
        torch.hub.load_state_dict_from_url(
            f"https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vit{model}14.pth",
            map_location="cpu",
        ),
        strict=True,
    )
    depth_anything.cpu()
    depth_anything.eval()



    import onnx
    from onnxsim import simplify
    RESOLUTION = [
            # [192,320],
        # [192,416],
        # [192,640],
        # [192,800],
            # [256,320],
        # [256,416],
        # [256,448],
        # [256,640],
        # [256,800],
        # [256,960],
        # [288,480],
        # [288,640],
        # [288,800],
        # [288,960],
        # [288,1280],
        # [320,320],
        # [384,480],
            # [384,640],
        # [384,800],
        # [384,960],
        # [384,1280],
        # [416,416],
        [490,644],
        # [480,800],
        # [480,960],
        # [480,1280],
        # [512,512],
        # [512,640],
            # [512,896],
        # [544,800],
        # [544,960],
        # [544,1280],
        # [640,640],
            # [736,1280],
    ]
    MODEL = f'depth_anything_vit{model}14'
    for H, W in RESOLUTION:
        onnx_file = f"{MODEL}_1x3x{H}x{W}.onnx"
        x = torch.randn(1, 3, H, W).cpu()
        torch.onnx.export(
            depth_anything,
            args=(x),
            f=onnx_file,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
        )
        model_onnx1 = onnx.load(onnx_file)
        model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
        onnx.save(model_onnx1, onnx_file)
        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)
        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)
        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)

        onnx_file = f"{MODEL}_Nx3x{H}x{W}.onnx"
        x = torch.randn(1, 3, H, W).cpu()
        torch.onnx.export(
            depth_anything,
            args=(x),
            f=onnx_file,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input' : {0: 'N'},
                'output' : {0: 'N', 1: '1', 2: f'{H}', 3: f'{W}'},
            }
        )
        model_onnx1 = onnx.load(onnx_file)
        model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
        onnx.save(model_onnx1, onnx_file)
        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)
        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)
        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)



    # onnx_file = f"{MODEL}_1x3xHxW.onnx"
    # x = torch.randn(1, 3, 192, 320).cpu()
    # torch.onnx.export(
    #     depth_anything,
    #     args=(x),
    #     f=onnx_file,
    #     opset_version=11,
    #     input_names=['input'],
    #     output_names=['output'],
    #     dynamic_axes={
    #         'input' : {2: 'height', 3: 'width'},
    #         'output' : {2: 'height', 3: 'width'},
    #     }
    # )
    # save_model(
    #     SymbolicShapeInference.infer_shapes(load_model(onnx_file), auto_merge=True),
    #     onnx_file,
    # )
    # model_onnx2 = onnx.load(onnx_file)
    # model_simp, check = simplify(model_onnx2)
    # onnx.save(model_simp, onnx_file)
    # model_onnx2 = onnx.load(onnx_file)
    # model_simp, check = simplify(model_onnx2)
    # onnx.save(model_simp, onnx_file)
    # model_onnx2 = onnx.load(onnx_file)
    # model_simp, check = simplify(model_onnx2)
    # onnx.save(model_simp, onnx_file)

    # onnx_file = f"{MODEL}_Nx3xHxW.onnx"
    # x = torch.randn(1, 3, 192, 320).cpu()
    # torch.onnx.export(
    #     depth_anything,
    #     args=(x),
    #     f=onnx_file,
    #     opset_version=11,
    #     input_names=['input'],
    #     output_names=['output'],
    #     dynamic_axes={
    #         'input' : {0: 'N', 2: 'height', 3: 'width'},
    #         'output' : {0: 'N', 2: 'height', 3: 'width'},
    #     }
    # )
    # save_model(
    #     SymbolicShapeInference.infer_shapes(load_model(onnx_file), auto_merge=True),
    #     onnx_file,
    # )
    # model_onnx2 = onnx.load(onnx_file)
    # model_simp, check = simplify(model_onnx2)
    # onnx.save(model_simp, onnx_file)
    # model_onnx2 = onnx.load(onnx_file)
    # model_simp, check = simplify(model_onnx2)
    # onnx.save(model_simp, onnx_file)
    # model_onnx2 = onnx.load(onnx_file)
    # model_simp, check = simplify(model_onnx2)
    # onnx.save(model_simp, onnx_file)


if __name__ == "__main__":
    args = parse_args()
    export_onnx(**vars(args))
