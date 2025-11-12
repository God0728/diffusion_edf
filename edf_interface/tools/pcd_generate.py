import sys
from pathlib import Path
import argparse
stereo_tools_path = Path("/home/hkcrc/handeye")
sys.path.insert(0, str(stereo_tools_path))

from StereoPCDTools.stereo_pcd_generator.test import generae_pcd_raw_images, generate_pcd_dir


def main():
    parse = argparse.ArgumentParser(description="pcd_generate")
    parse.add_argument("--img",default="/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/img2")
    parse.add_argument("--camera_model",default="/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/configs/realsense_camera_model.json")
    parse.add_argument("--output",default="../pcd_test")
    args = parse.parse_args()
    img_dir = Path(args.img)

    if img_dir.is_dir():
        generate_pcd_dir(
            raw_dir=args.img,
            camera_model_path=args.camera_model,
            output_dir=args.output,
            scale=1
        )
    elif img_dir.is_file():
        if "A_" in args.img:
            raw_left_path = args.img
            raw_right_path = args.img.replace("A_", "D_")
        elif "D_" in args.img:
            raw_left_path = args.img.replace("D_", "A_")
            raw_right_path = args.img
        generae_pcd_raw_images(
            raw_left_path=raw_left_path,
            raw_right_path=raw_right_path,
            camera_model_path=args.camera_model,
            output_dir=args.output,
            scale=1
        )

if __name__ == "__main__":
    main()

