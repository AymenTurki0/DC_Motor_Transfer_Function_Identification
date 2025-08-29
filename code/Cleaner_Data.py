import pandas as pd
import numpy as np
import os

def clean_and_scale_motor_data(input_file, output_file, pwm_value=175, dt_ms=10):
    # Load the CSV file
    df = pd.read_csv(input_file)

    # --- Remove leading zeros ---
    first_nonzero_idx = df[df["speed_rpm"] != 0].index.min()

    # --- Remove trailing zeros ---
    last_nonzero_idx = df[df["speed_rpm"] != 0].index.max()

    # Keep one initial zero, then crop
    cleaned_df = pd.concat(
        [df.iloc[[0]], df.iloc[first_nonzero_idx:last_nonzero_idx+1]]
    ).reset_index(drop=True)

    # --- Step 1: Rebuild time axis with fixed sampling ---
    N = len(cleaned_df)
    dt = dt_ms / 1000.0  # seconds
    cleaned_df["time_s"] = np.arange(N) * dt

    # --- Step 2: Extract speed (RPM) ---
    y_rpm = cleaned_df["speed_rpm"].values

    # --- Step 3: Normalize input and output ---
    u_norm = np.ones_like(y_rpm, dtype=float) * (pwm_value / 255.0)
    y_norm = y_rpm / np.max(y_rpm)

    cleaned_df["input_norm"] = u_norm
    cleaned_df["output_norm"] = y_norm

    # --- Step 4: Save cleaned + scaled file ---
    cleaned_df.to_csv(output_file, index=False)
    print(f"✅ Cleaned and scaled data saved to: {output_file}")


if __name__ == "__main__":
    # Get project root (one level above "code")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Define input/output paths relative to project root
    input_file = os.path.join(project_root, "data_initial", "encoder_data_25.csv")
    output_file = os.path.join(project_root, "data_cleaned", "Scaled_encoder_data_255.csv")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Run cleaning + scaling
    clean_and_scale_motor_data(input_file, output_file)
