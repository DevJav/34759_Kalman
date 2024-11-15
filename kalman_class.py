import numpy as np

class KalmanFilter:
    dt = 1.0

    # The initial state. The robot starts in position 0 with the velocity 0.
    x = np.array([[0], # Position along the x-axis
                [0], # Velocity along the x-axis
                [0], # Position along the y-axis
                [0], # Velocity along the y-axis
                [0], # Position along the z-axis
                [0]]) # Velocity along the z-axis

    # The initial uncertainty. We start with some very large values.
    P = np.array([[1000, 0, 0, 0, 0, 0],
                [0, 1000, 0, 0, 0, 0],
                [0, 0, 1000, 0, 0, 0],
                [0, 0, 0, 1000, 0, 0],
                [0, 0, 0, 0, 1000, 0],
                [0, 0, 0, 0, 0, 1000]])

    # The external motion. Set to 0 here.
    u = np.array([[0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0]])

    # The transition matrix. 
    F = np.array([[1, dt, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, dt, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 1]])

    # The observation matrix. We only get the position as measurement.
    H = np.array([[1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0]])

    # The measurement uncertainty
    R = np.diag([20, 20, 20]) 

    # The identity matrix. Simply a matrix with 1 in the diagonal and 0 elsewhere.
    I = np.array([[1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]])
    
    def __init__(self, id, x, y, z, object_type, color='blue'):
        self.id = id
        self.x[0] = x
        self.x[2] = y
        self.x[4] = z
        self.object_type = object_type
        self.color = color

    def update(self, Z):
        ### Insert update function
        y = Z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), np.transpose(self.H)) + self.R
        K = np.dot(np.dot(self.P, np.transpose(self.H)), np.linalg.pinv(S))
        x_p = self.x + np.dot(K, y)
        P_p = np.dot((self.I - np.dot(K, self.H)), self.P)

        self.x = x_p
        self.P = P_p

    def predict(self):
        ### insert predict function
        x_p = np.dot(self.F, self.x) + self.u
        P_p = np.dot(np.dot(self.F, self.P), np.transpose(self.F))

        self.x = x_p
        self.P = P_p

    def get_location(self):
        return self.x[0].item(), self.x[2].item(), self.x[4].item()

    def __repr__(self) -> str:
        return f'ID: {self.id}, Location: {self.get_location()}, Object Type: {self.object_type}'