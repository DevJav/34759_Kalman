import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
from kalman_class import KalmanFilter

MIX_X = 0
MAX_X = 0
MIN_Z = 0
MAX_Z = 0

def get_labels():
    labels_path = '34759_final_project_rect/seq_01/'
    labels_file = 'labels.txt'

    with open(labels_path + labels_file, 'r') as f:
        lines = f.readlines()

    # Create a list of all the labels
    labels = []
    for line in lines:
        labels.append(line.split())

    return labels

def get_frame(label):
    return int(label[0])

def get_location(label):
    x, y, z = float(label[13]), float(label[14]), float(label[15])
    return x, y, z

def get_id(label):
    return int(label[1])

def get_object_type(label):
    return label[2]

def plot_point(ax, x, z, color, track_id, object_type, alpha=1.0):
    """Plots a point with text annotation on the specified axis."""
    marker = {'Car': 'x', 'Pedestrian': 'o', 'Cyclist': '^'}.get(object_type, 'o')
    ax.plot(x, z, color=color, marker=marker, alpha=alpha)
    ax.text(x, z, track_id, fontsize=12, color=color)

def get_min_max_x_z(labels):
    min_x, max_x = 0, 0
    min_z, max_z = 0, 0
    for label in labels:
        x, y, z = get_location(label)
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if z < min_z:
            min_z = z
        if z > max_z:
            max_z = z
    return min_x, max_x, min_z, max_z


def plot_points(ax1, ax2):
    ax1.set_title('Ground Truth')
    ax1.set_xlim(MIX_X, MAX_X)
    ax1.set_ylim(MIN_Z, MAX_Z)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.grid(True)
    ax2.set_title('Kalman Filter')
    ax2.set_xlim(MIX_X, MAX_X)
    ax2.set_ylim(MIN_Z, MAX_Z)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.grid(True)
    clear_output(wait=True)
    plt.pause(0.1)
    ax1.clear()
    ax2.clear()

if __name__ == '__main__':

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('grey')
    ax1.set_facecolor('grey')
    ax2.set_facecolor('grey')

    tracked_objects = {}

    labels = get_labels()

    MIX_X, MAX_X, MIN_Z, MAX_Z = get_min_max_x_z(labels)

    current_frame = 0
    active_objects = []
    for label in labels:
        if int(get_frame(label)) != current_frame:
            for id, obj in tracked_objects.items():
                if id not in active_objects:
                    obj.predict()
                    plot_point(ax2, obj.x[0], obj.x[4], 'red', id, obj.object_type)

            plot_points(ax1, ax2)
            current_frame += 1
            active_objects = []

        id = get_id(label)
        object_type = get_object_type(label)
        x, y, z = get_location(label)
        if id not in tracked_objects:
            tracked_objects[id] = KalmanFilter(id, x, y, z, object_type)
        else:
            tracked_objects[id].predict()
            tracked_objects[id].update(np.array([[x], [y], [z]]))
        plot_point(ax1, x, z, 'blue', id, object_type)
        plot_point(ax2, tracked_objects[id].x[0], tracked_objects[id].x[4], 'green', id, object_type)

        active_objects.append(id)
