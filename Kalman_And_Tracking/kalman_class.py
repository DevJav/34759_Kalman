import numpy as np

class KalmanFilter:
    
    def __init__(self, id, x, y, z, object_type, color='blue'):
        self.dt = 0.1

        # The initial state. The robot starts in position 0 with the velocity 0.
        self.x = np.array([[0], # Position along the x-axis
                    [0], # Velocity along the x-axis
                    [0], # Position along the y-axis
                    [0], # Velocity along the y-axis
                    [0], # Position along the z-axis
                    [0]]) # Velocity along the z-axis

        # The initial uncertainty. We start with some very large values.
        self.initial_uncertainty = 10
        self.P = np.array([[self.initial_uncertainty, 0, 0, 0, 0, 0],
                    [0, self.initial_uncertainty, 0, 0, 0, 0],
                    [0, 0, self.initial_uncertainty, 0, 0, 0],
                    [0, 0, 0, self.initial_uncertainty, 0, 0],
                    [0, 0, 0, 0, self.initial_uncertainty, 0],
                    [0, 0, 0, 0, 0, self.initial_uncertainty]])

        # The external motion. Set to 0 here.
        self.u = np.array([[0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0]])

        # The transition matrix. 
        self.F = np.array([[1, self.dt, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, self.dt, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, self.dt],
                    [0, 0, 0, 0, 0, 1]])

        # The observation matrix. We only get the position as measurement.
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0]])

        # The measurement uncertainty
        self.R = np.diag([20, 20, 20]) 

        # The identity matrix. Simply a matrix with 1 in the diagonal and 0 elsewhere.
        self.I = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]])
        self.id = id
        self.x[0] = x
        self.x[2] = y
        self.x[4] = z
        self.object_type = object_type
        self.color = color

        self.disconnect_count = 0

    def update(self, Z):
        ### Insert update function
        y = Z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), np.transpose(self.H)) + self.R
        K = np.dot(np.dot(self.P, np.transpose(self.H)), np.linalg.pinv(S))
        x_p = self.x + np.dot(K, y)
        P_p = np.dot((self.I - np.dot(K, self.H)), self.P)

        self.x = x_p
        self.P = P_p

        self.disconnect_count = 0

    def predict(self):
        ### insert predict function
        x_p = np.dot(self.F, self.x) + self.u
        P_p = np.dot(np.dot(self.F, self.P), np.transpose(self.F))

        self.x = x_p
        self.P = P_p

        self.disconnect_count += 1

    def get_location(self):
        return self.x[0].item(), self.x[2].item(), self.x[4].item()

    def __repr__(self) -> str:
        return f'ID: {self.id}, Location: {self.get_location()}, Object Type: {self.object_type}'
    
    def get_covariance(self):
        return self.P