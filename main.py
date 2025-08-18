def main():
    print("Hello from figshare!")


def run_test():
    pass


def clean_results():
    """Clean all result data"""
    import os
    import glob
    import shutil
    
    # Define file patterns to delete
    patterns = [
        "1-Ball_Calibration/*.jpg",
        "1-Ball_Calibration/*pkl",
        "1-Ball_Calibration/*.mat",
        "1-Ball_Calibration/*keypoints.mp4",
        "2-Social_3D_Pose/*segpkl",
        "2-Social_3D_Pose/*matcalibpkl",
        "2-Social_3D_Pose/*com3d*.mp4",
        "2-Social_3D_Pose/*mask*.mp4",
        "2-Social_3D_Pose/*smoothed*.mp4",
        "2-Social_3D_Pose/*400p*.mp4",
        "3-Sequence_Labeling/*.pkl",
    ]
    
    deleted_count = 0
    
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {file}: {e}")
    
    # Remove output directory if it exists
    for output_dir in ["output", "3-Sequence_Labeling/out_semiseq2seq_norm"]:
        if os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
                print(f"Deleted directory: {output_dir}")
            except Exception as e:
                print(f"Failed to delete directory {output_dir}: {e}")
    
    print(f"Total files deleted: {deleted_count}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            success = run_test()
            sys.exit(0 if success else 1)
        elif sys.argv[1] == "clean":
            clean_results()
        else:
            print("Usage: python main.py [test|clean]")
            print("  test: Run all tests to generate figures and check if they are correct")
            print("  clean: Clean all result files")
    else:
        main()
