from loguru import logger
import time
from edf_interface.modules.camera import RealSenseCalibrator
import argparse


def main():
    parser = argparse.ArgumentParser(description="capture_img")
    parser.add_argument("-d", "--device",  default='grasp')
    parser.add_argument("--output-dir", default="/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/img2")
    parser.add_argument('-s', action='store_true')    
    args = parser.parse_args()

    calibrator = RealSenseCalibrator(args.device)

    logger.info("waiting..")
    time.sleep(2)
    
    if args.s:
        calibrator.save_json()

    calibrator.capture_images(output_dir=args.output_dir)
        
    calibrator.stop()

if __name__ == "__main__":
    main()