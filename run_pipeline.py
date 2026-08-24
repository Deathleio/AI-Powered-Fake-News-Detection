import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="AI-Powered Fake News Detection Pipeline")
    parser.add_argument("--mode", choices=["train", "serve", "test"], default="train", help="Pipeline execution mode")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        from train import run_training_pipeline
        run_training_pipeline()
    elif args.mode == "serve":
        from src.serving.app import launch_server
        launch_server(host=args.host, port=args.port)
    elif args.mode == "test":
        import unittest
        loader = unittest.TestLoader()
        suite = loader.discover("tests")
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)

if __name__ == '__main__':
    main()
