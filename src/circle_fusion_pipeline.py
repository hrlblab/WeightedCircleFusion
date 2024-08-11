import argparse
import os
import subprocess


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run a Python script with multiple model paths.")
    parser.add_argument("--models_folder", required=True, help="Path to the folder containing the model files.")
    parser.add_argument("--other_args", nargs=argparse.REMAINDER, help="Other arguments to pass to the script.")

    # Parse arguments
    args = parser.parse_args()

    # List all model files in the specified folder
    model_paths = [os.path.join(args.models_folder, model_file) for model_file in os.listdir(args.models_folder) if
                   model_file.endswith('.pth')]

    # Define the base command, excluding the model path
    base_command = [
        "python", "/data/CircleNet/src/run_detection_for_scn.py",  # Adjust this path as needed
    ]

    # Include other arguments from the command line
    base_command.extend(args.other_args)

    # Loop through each model path and run the command with each model
    for model_path in model_paths:
        # Construct the full command with the current model path
        command = base_command + ["--load_model", model_path]

        # Execute the command
        print("Running command:", " ".join(command))
        result = subprocess.run(command, capture_output=True, text=True)

        # Check if there was an error
        if result.returncode != 0:
            print(f"Error running command with model {model_path}: {result.stderr}")
        else:
            print(f"Command completed successfully for model {model_path}")


if __name__ == "__main__":
    main()
