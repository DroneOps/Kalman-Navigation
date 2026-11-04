"""
Script to read Y and Z error values and plot them against the time steps.

Author: DroneOps
Repository: Kalman-Navigation
"""

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# Script to read Y and Z error values and plot them against the time steps
def plot_errors(file_path):
    
    # Read the data from the file
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    
    # Extract time steps, Y errors, and Z errors
    time_steps = data[:, 0]
    y_errors = data[:, 1]
    z_errors = data[:, 2]
    
    # Create a figure and axis for plotting
    plt.figure(figsize=(10, 6))
    
    # Plot Y errors
    plt.plot(time_steps, y_errors, label='Y Error', color='blue')
    
    # Plot Z errors
    plt.plot(time_steps, z_errors, label='Z Error', color='orange')
    
    # Add labels and title
    plt.xlabel('Time Steps')
    plt.ylabel('Error Values')
    plt.title('Y and Z Errors Over Time')
    
    # Add legend
    plt.legend()
    
    # Show grid
    plt.grid()
    
    # Display the plot
    plt.show()

def main():
    # Path to the CSV file containing the error data
    file_path = 'error_data.csv'  # Update this path to your actual file location
    
    # Call the function to plot errors
    plot_errors(file_path)

if __name__ == "__main__":
    main()