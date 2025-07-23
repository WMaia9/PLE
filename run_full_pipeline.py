import time
# Import the functions from the correctly named modules
from a1_calculate_g_factor import run_g_factor_calculation
from a2_process_sample import run_sample_analysis


def main():
    """
    Runs the full analysis pipeline:
    1. Calculates the G-factor.
    2. Processes the sample using the calculated G-factor.
    """
    print("🚀 Starting full analysis pipeline...")
    
    # Step 1: Calibration
    print("\n--- Step 1: Calculating G-Factor ---")
    start_time = time.time()
    run_g_factor_calculation()
    print(f"--- Step 1 finished in {time.time() - start_time:.2f} seconds ---\n")
    
    # Step 2: Sample Analysis
    print("--- Step 2: Analyzing Sample ---")
    start_time = time.time()
    run_sample_analysis()
    print(f"--- Step 2 finished in {time.time() - start_time:.2f} seconds ---\n")
    
    print("✅ Full pipeline finished successfully!")


if __name__ == "__main__":
    main()