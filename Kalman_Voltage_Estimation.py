import numpy as np
import matplotlib.pyplot as plt
from Kalman_Filter import general_kalman_filter, plot_kalman_results

def simulate_voltage_measurements(t, u=14.4, s=4):
    """Simulate noisy voltage measurements."""
    return [np.random.randn() * s + u for _ in t]

def main():
    """Main function to run Kalman filter on voltage measurements."""
    t = np.linspace(1, 1000, num=100)
    z_measurements = simulate_voltage_measurements(t)

    A = np.array([[1]])  # State transition matrix for voltage (1D)
    H = np.array([[1]])  # Measurement matrix for voltage (1D)
    Q = np.array([[0]])  # Process noise covariance for voltage (1D)
    R = np.array([[4.0]])  # Measurement noise covariance for voltage (1D)
    x_init = np.array([14])  # Initial state estimate
    P_init = np.array([[6]])  # Initial covariance estimate

    estimates, p_error = general_kalman_filter(z_measurements, A, H, Q, R, x_init, P_init)
    plot_kalman_results(estimates, z_measurements, p_error, ['Voltage'])

if __name__ == "__main__":
    main()
