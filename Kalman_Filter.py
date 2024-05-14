import numpy as np
import matplotlib.pyplot as plt

def general_kalman_filter(z_measurements, A, H, Q, R, x_init, P_init):
    """Apply Kalman filter to estimate states over time.

    Args:
        z_measurements (list or numpy.ndarray): List of measurements.
        A (numpy.ndarray): State transition matrix.
        H (numpy.ndarray): Measurement matrix.
        Q (numpy.ndarray): Process noise covariance.
        R (numpy.ndarray): Measurement noise covariance.
        x_init (numpy.ndarray): Initial state estimate.
        P_init (numpy.ndarray): Initial covariance estimate.

    Returns:
        tuple: Lists of state estimates and error covariances.
    """
    x = x_init
    P = P_init

    estimates = []
    p_error = []

    for z in z_measurements:
        # Time Update (Prediction)
        x = A @ x  # State prediction
        P = A @ P @ A.T + Q  # Covariance prediction

        # Measurement Update (Correction)
        S = H @ P @ H.T + R  # Innovation (residual) covariance
        K = P @ H.T @ np.linalg.inv(S)  # Kalman Gain
        x = x + K @ (z - H @ x)  # State update
        P = (np.eye(len(x)) - K @ H) @ P  # Covariance update

        # Store the estimates
        estimates.append(x.copy())
        p_error.append(np.diag(P))

    return estimates, p_error

def plot_kalman_results(estimates, measurements, p_error, labels):
    """Plot Kalman filter results."""
    estimates = np.array(estimates)
    p_error = np.array(p_error)
    
    plt.figure(figsize=(12, 10))

    # Plot each state estimate and its corresponding measurements and errors
    for i, label in enumerate(labels):
        plt.subplot(len(labels), 1, i + 1)
        plt.plot(measurements, label='Measurements', marker='o')
        plt.plot(estimates[:, i], label=f'Kalman Filter Estimates ({label})', linestyle='--')
        plt.fill_between(range(len(measurements)), 
                         estimates[:, i] - np.sqrt(p_error[:, i]),
                         estimates[:, i] + np.sqrt(p_error[:, i]),
                         color='yellow', alpha=0.3, label=f'{label} error')
        plt.title(f'Kalman Filter Estimation vs Measurements ({label})')
        plt.xlabel('Time Step')
        plt.ylabel(label)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_kalman_results(estimates, measurements, p_error, labels):
    """Plot Kalman filter results."""
    estimates = np.array(estimates)
    p_error = np.array(p_error)
    
    plt.figure(figsize=(12, 10))

    # Plot each state estimate and its corresponding measurements and errors
    for i, label in enumerate(labels):
        plt.subplot(len(labels), 1, i + 1)
        plt.plot(measurements, label='Measurements', marker='o')
        plt.plot(estimates[:, i], label=f'Kalman Filter Estimates ({label})', linestyle='--')
        plt.fill_between(range(len(measurements)), 
                         estimates[:, i] - np.sqrt(p_error[:, i]),
                         estimates[:, i] + np.sqrt(p_error[:, i]),
                         color='yellow', alpha=0.3, label=f'{label} error')
        plt.title(f'Kalman Filter Estimation vs Measurements ({label})')
        plt.xlabel('Time Step')
        plt.ylabel(label)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()
