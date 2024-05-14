import numpy as np
import matplotlib.pyplot as plt
from Kalman_Filter import general_kalman_filter, plot_kalman_results


def simulate_sonar_measurements(t):
    """Simulate noisy sonar measurements."""
    return [5 * np.log(1 + tt * 10) + np.random.randn() * 2 + 3 for tt in t]

def main():
    """Main function to run Kalman filter on sonar measurements."""
    dt = 0.05
    t = np.arange(0, 10, dt)
    z_measurements = simulate_sonar_measurements(t)

    A = np.array([[1, dt], [0, 1]])  # State transition matrix for sonar (2D)
    H = np.array([[1, 0]])  # Measurement matrix for sonar (2D)
    Q = np.array([[1, 0], [0, 3]])  # Process noise covariance for sonar (2D)
    R = np.array([[10]])  # Measurement noise covariance for sonar (1D)
    x_init = np.array([0, 0])  # Initial state estimate (position and velocity)
    P_init = np.array([[1000, 0], [0, 1000]])  # Initial covariance estimate

    estimates, p_error = general_kalman_filter(z_measurements, A, H, Q, R, x_init, P_init)
    plot_kalman_results(estimates, z_measurements, p_error, ['Position', 'Velocity'])
    
if __name__ == "__main__":
    main()
