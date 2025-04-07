"""Entry point for Posture & Emotions detection."""

from loguru import logger
import os
import time
import tempfile
import argparse

from src import (
    Source,
    Grabber,
    Posture,
    Plotter,
    Emotion,
    download_artifact
)


def run(args):
    """runs actual processes."""
    with tempfile.TemporaryDirectory() as tmp_dir:

        if args.path:
            models_path = args.path
        else:
            models_path = download_artifact(path=tmp_dir)

        assert os.path.exists(models_path), "models not found!!"
    
        grabber = Grabber(Source.Camera, "")
        pose = Posture(models_path)
        emotion = Emotion(models_path)

        if args.display:
            plot = Plotter()

        fc = 0 # frame count
        st = time.time()  # start time
        while grabber.is_open():
            ct = time.time()
            frame = grabber.get()
            fc += 1
            post, emot, conf= 0, 0, 0
            if pose:
                post, kpts = pose(frame)
            if emotion:
                emot, conf = emotion(frame, kpts)
            if args.display and plot:
                k = plot.show(frame, kpts, [post, emot])
                if k == 27:
                    break
            msg = ""
            if post and emot:
                msg = f"Postion: {post}; Emotion: {emot}({conf:.2f}))"
            d = time.time() - st
            if d >= 3:
                fps = fc / d
                st = time.time()
                fc = 0
                msg += f" FPS: {fps:.2f}"
            if msg:
                logger.info(msg)

            # maintain 15 FPS
            d = time.time() - ct
            if d < 0.066:
                time.sleep(0.066-d)

        grabber.close()
        logger.info("application closed successfully.")

def main():
    """main driver function."""
    parser = argparse.ArgumentParser("Posture & Expression detection.")
    parser.add_argument("-d", "--display", action="store_true", help="Flag to diplay output.")
    parser.add_argument("-v", "--version", action="store_true", help="Flag to diplay version.")
    parser.add_argument("-p", "--path", type=str, help="Provide model path.")
    args = parser.parse_args()
    if args.version:
        print ("Major Version: 1.0.0")
        print ("Posture Version: 0.0.1")
        print ("Emotion Version: 0.0.1")
    else:
        run(args)

if __name__=="__main__":
    main()
