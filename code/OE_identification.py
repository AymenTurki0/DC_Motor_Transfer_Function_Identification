import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.optimize import minimize

def plot_motor_response(csv_file_path):
    """
    Plot motor response showing RPM vs Time for setpoint 255
    
    Parameters:
    csv_file_path (str): Path to the CSV file
    """
    
    try:
        # Read the CSV file
        # Assuming the CSV has headers: time_ms, speed_rpm, time_s, input_norm, output_norm
        df = pd.read_csv(csv_file_path)
        
        # Clean column names (remove whitespace)
        df.columns = df.columns.str.strip()
        
        # Display basic info about the data
        print("Data loaded successfully!")
        print(f"Number of data points: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot RPM vs Time (using time_ms for milliseconds) - THIS IS THE IMPORTANT PART
        plt.plot(df['time_ms'], df['speed_rpm'], 'b-', linewidth=2, label='Motor Speed')
        
        # Add setpoint reference line at 255 RPM
        plt.axhline(y=150, color='r', linestyle='--', linewidth=2, label='Setpoint (150 RPM)')
        
        # Customize the plot
        plt.xlabel('Time (milliseconds)', fontsize=12)  # X-AXIS IS IN MILLISECONDS
        plt.ylabel('Speed (RPM)', fontsize=12)
        plt.title('Motor Response to Setpoint 150 RPM', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Set y-axis limit to reach 1300 RPM
        plt.ylim(0, 1300)
        
        # Add some statistics
        max_speed = df['speed_rpm'].max()
        min_speed = df['speed_rpm'].min()
        final_speed = df['speed_rpm'].iloc[-1]
        
        # Add text box with statistics
        stats_text = f'Max Speed: {max_speed:.1f} RPM\nMin Speed: {min_speed:.1f} RPM\nFinal Speed: {final_speed:.1f} RPM'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Improve layout
        plt.tight_layout()
        
        # Save the plot as 255.png
        plt.savefig('150.png', dpi=300, bbox_inches='tight')
        print("Plot saved as '150.png'")
        
        # Show the plot
        plt.show()
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{csv_file_path}'")
        print("Please make sure the file path is correct.")
        return None
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None

def error_output_identification(input_signal, output_signal, dt):
    """
    System identification using Error Output (EO) method
    Identifies transfer function G(z) = (b1*z + b2) / (z^2 + a1*z + a2)
    
    Parameters:
    input_signal: Input signal (normalized)
    output_signal: Output signal (RPM)
    dt: Sampling time in seconds
    
    Returns:
    a1, a2, b1, b2: Transfer function coefficients
    """
    
    # Prepare data for identification
    N = len(output_signal)
    
    # Create the regression matrix (Phi) and output vector (Y)
    # For second-order system: y(k) = -a1*y(k-1) - a2*y(k-2) + b1*u(k-1) + b2*u(k-2)
    
    # We need at least 3 data points for second-order system
    if N < 3:
        raise ValueError("Not enough data points for identification")
    
    # Build regression matrix Phi and output vector Y
    Phi = []
    Y = []
    
    for k in range(2, N):  # Start from k=2 to have y(k-1) and y(k-2)
        phi_row = [-output_signal[k-1], -output_signal[k-2], input_signal[k-1], input_signal[k-2]]
        Phi.append(phi_row)
        Y.append(output_signal[k])
    
    Phi = np.array(Phi)
    Y = np.array(Y)
    
    # Solve using least squares: theta = (Phi^T * Phi)^-1 * Phi^T * Y
    try:
        theta = np.linalg.lstsq(Phi, Y, rcond=None)[0]
        a1, a2, b1, b2 = theta
        
        return a1, a2, b1, b2
        
    except np.linalg.LinAlgError:
        print("Error in least squares solution")
        return None, None, None, None

def validate_transfer_function(input_signal, output_signal, a1, a2, b1, b2, dt):
    """
    Validate the identified transfer function by simulating the response
    
    Parameters:
    input_signal: Input signal
    output_signal: Actual output signal
    a1, a2, b1, b2: Transfer function coefficients
    dt: Sampling time
    
    Returns:
    simulated_output: Simulated output using identified transfer function
    fit_percentage: Fit percentage
    """
    
    # Create discrete transfer function
    num = [b1, b2]
    den = [1, a1, a2]
    
    # Simulate the system response
    _, simulated_output = signal.dlsim((num, den, dt), input_signal)
    simulated_output = simulated_output.flatten()
    
    # Calculate fit percentage
    # Fit = (1 - ||y - y_sim|| / ||y - mean(y)||) * 100
    y_mean = np.mean(output_signal)
    numerator = np.linalg.norm(output_signal - simulated_output)
    denominator = np.linalg.norm(output_signal - y_mean)
    
    if denominator != 0:
        fit_percentage = (1 - numerator/denominator) * 100
    else:
        fit_percentage = 0
    
    return simulated_output, fit_percentage

def system_identification_analysis(df):
    """
    Perform complete system identification analysis
    
    Parameters:
    df: DataFrame with motor data
    """
    
    print("\n" + "="*60)
    print("SYSTEM IDENTIFICATION ANALYSIS")
    print("="*60)
    
    # Extract signals
    time_ms = df['time_ms'].values
    input_signal = df['input_norm'].values  # Normalized input
    output_signal = df['speed_rpm'].values  # RPM output
    
    # Calculate sampling time (convert ms to seconds)
    dt = (time_ms[1] - time_ms[0]) / 1000.0  # Convert ms to seconds
    print(f"Sampling time: {dt:.4f} seconds ({time_ms[1] - time_ms[0]:.1f} ms)")
    
    # Perform system identification
    print("\nPerforming Error-Output (EO) identification...")
    a1, a2, b1, b2 = error_output_identification(input_signal, output_signal, dt)
    
    if a1 is not None:
        print(f"\nIdentified Transfer Function G(z):")
        print(f"G(z) = ({b1:.6f}*z + {b2:.6f}) / (z² + {a1:.6f}*z + {a2:.6f})")
        print(f"\nCoefficients:")
        print(f"a1 = {a1:.6f}")
        print(f"a2 = {a2:.6f}")
        print(f"b1 = {b1:.6f}")
        print(f"b2 = {b2:.6f}")
        
        # Validate the transfer function
        print("\nValidating transfer function...")
        simulated_output, fit_percentage = validate_transfer_function(
            input_signal, output_signal, a1, a2, b1, b2, dt
        )
        
        print(f"Model fit: {fit_percentage:.2f}%")
        
        # Plot comparison
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Time response comparison
        plt.subplot(2, 1, 1)
        plt.plot(time_ms, output_signal, 'b-', linewidth=2, label='Measured Output')
        plt.plot(time_ms, simulated_output, 'r--', linewidth=2, label='Simulated Output')
        plt.xlabel('Time (ms)')
        plt.ylabel('Speed (RPM)')
        plt.title(f'Model Validation - Fit: {fit_percentage:.2f}%')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Error signal
        plt.subplot(2, 1, 2)
        error = output_signal - simulated_output
        plt.plot(time_ms, error, 'g-', linewidth=1, label='Error')
        plt.xlabel('Time (ms)')
        plt.ylabel('Error (RPM)')
        plt.title('Model Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('system_identification_validation2.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyze poles and zeros
        print(f"\nSystem Analysis:")
        poles = np.roots([1, a1, a2])
        zeros = np.roots([b1, b2]) if b1 != 0 or b2 != 0 else []
        
        print(f"Poles: {poles}")
        if len(zeros) > 0:
            print(f"Zeros: {zeros}")
        
        # Check stability
        pole_magnitudes = np.abs(poles)
        if np.all(pole_magnitudes < 1.0):
            print("System is STABLE (all poles inside unit circle)")
        else:
            print("System is UNSTABLE (poles outside unit circle)")
        
        return a1, a2, b1, b2, fit_percentage
    
    else:
        print("System identification failed!")
        return None, None, None, None, None

# Usage example
if __name__ == "__main__":
    # Your exact file path - DO NOT CHANGE THIS
    csv_file_path = r'C:\Users\LENOVO\Desktop\Motor_Transfer_Function_Identification\data_cleaned\Cleaner_encoder_data_150.csv'
    
    # Plot the motor response
    data = plot_motor_response(csv_file_path)
    
    # Perform system identification analysis
    if data is not None:
        print(f"\nAdditional Analysis:")
        print(f"Test duration: {data['time_ms'].max():.0f} milliseconds")  # USES time_ms COLUMN
        print(f"Average speed: {data['speed_rpm'].mean():.1f} RPM")
        print(f"Speed standard deviation: {data['speed_rpm'].std():.1f} RPM")
        
        # Run system identification
        a1, a2, b1, b2, fit = system_identification_analysis(data)
        
        if a1 is not None:
            print(f"\n" + "="*60)
            print("FINAL RESULTS:")
            print("="*60)
            print(f"Transfer Function: G(z) = ({b1:.6f}*z + {b2:.6f}) / (z² + {a1:.6f}*z + {a2:.6f})")
            print(f"Model Fit: {fit:.2f}%")
            print("Files saved: '150.png' and 'system_identification_validation.png'")
            print("="*60)