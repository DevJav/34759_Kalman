import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
from kalman_class import KalmanFilter
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

MIX_X = 0
MAX_X = 0
MIN_Z = 0
MAX_Z = 0

DISTANCE_THRESHOLD = 5
DISCONNECT_THRESHOLD = 10

colors = {
    '0': (255, 0, 0), '1': (0, 255, 0), '2': (0, 0, 255), '3': (255, 255, 0), '4': (255, 0, 255),
    '5': (0, 255, 255), '6': (128, 0, 0), '7': (0, 128, 0), '8': (0, 0, 128), '9': (128, 128, 0),
    '10': (128, 0, 128), '11': (0, 128, 128), '12': (192, 192, 192), '13': (128, 128, 128),
    '14': (64, 0, 0), '15': (0, 64, 0), '16': (0, 0, 64), '17': (64, 64, 0), '18': (64, 0, 64),
    '19': (0, 64, 64), '20': (255, 128, 0), '21': (255, 0, 128), '22': (128, 255, 0), '23': (0, 255, 128),
    '24': (128, 0, 255), '25': (0, 128, 255), '26': (255, 128, 128), '27': (128, 255, 128),
    '28': (128, 128, 255), '29': (255, 255, 128), '30': (255, 128, 255), '31': (128, 255, 255),
    '32': (64, 128, 128), '33': (128, 64, 128), '34': (128, 128, 64), '35': (64, 64, 128),
    '36': (64, 128, 64), '37': (128, 64, 64), '38': (64, 64, 64), '39': (192, 64, 64),
    '40': (64, 192, 64), '41': (64, 64, 192), '42': (192, 192, 64), '43': (192, 64, 192),
    '44': (64, 192, 192), '45': (192, 128, 64), '46': (192, 64, 128), '47': (128, 192, 64),
    '48': (64, 128, 192), '49': (128, 64, 192), '50': (192, 128, 128), '51': (128, 192, 128),
    '52': (128, 128, 192), '53': (192, 192, 128), '54': (192, 128, 192), '55': (128, 192, 192)
}


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

def get_object_type(label):
    return label[2]

def plot_point(ax, x, y, z, color, object_type, track_id=-1, alpha=1.0):
    """Plots a point with text annotation on the specified axis."""
    marker = {'Car': 'x', 'Pedestrian': 'o', 'Cyclist': '^'}.get(object_type, 'o')
    ax.scatter(x, z, y, color=color, marker=marker, alpha=alpha)
    if track_id != -1:
        ax.text(x, z, y, track_id, fontsize=12, color=color, alpha=alpha)

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
    ax1.set_zlim(MIN_Z, MAX_Z)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.grid(True)
    ax2.set_title('Kalman Filter')
    ax2.set_xlim(MIX_X, MAX_X)
    ax2.set_ylim(MIN_Z, MAX_Z)
    ax2.set_zlim(MIN_Z, MAX_Z)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.grid(True)
    clear_output(wait=True)
    plt.pause(0.25)
    ax1.clear()
    ax2.clear()

if __name__ == '__main__':

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    fig.patch.set_facecolor('grey')
    ax1.set_facecolor('grey')
    ax2.set_facecolor('grey')

    track_index = 0
    tracked_objects = {}

    labels = get_labels()

    MIX_X, MAX_X, MIN_Z, MAX_Z = get_min_max_x_z(labels)
    print(MIX_X, MAX_X, MIN_Z, MAX_Z)

    current_frame = 0

    object_locations = []
    object_types = []

    for label in labels:
        if int(get_frame(label)) != current_frame:

            # predict all objects
            for id, obj in tracked_objects.items():
                obj.predict()

            # calculate distance of all objects to all kalman filters
            distances = {}
            for object_idx, object in enumerate(object_locations):
                for kalman_idx, kalman in tracked_objects.items():
                    distance = np.linalg.norm(np.array(object) - np.array(kalman.get_location()))
                    distances[(object_idx, kalman_idx)] = float(distance)

            # Create a cost matrix
            num_objects = len(object_locations)
            num_kalman_filters = len(tracked_objects)
            cost_matrix = np.full((num_objects, num_kalman_filters), np.inf)  # Default to large costs

            # Fill in the cost matrix using your distance calculations
            for i, object_loc in enumerate(object_locations):
                for j, kalman in enumerate(tracked_objects.values()):
                    cost_matrix[i, j] = np.linalg.norm(np.array(object_loc) - np.array(kalman.get_location()))

            # Solve the assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Filter matches by the distance threshold
            matches = []
            unmatched_objects = set(range(num_objects))
            unmatched_kalman = set(tracked_objects.keys())


            for obj_idx, kalman_idx in zip(row_ind, col_ind):
                if cost_matrix[obj_idx, kalman_idx] <= DISTANCE_THRESHOLD:
                    matches.append((obj_idx, kalman_idx))
                    unmatched_objects.discard(obj_idx)
                    unmatched_kalman.discard(kalman_idx)

            # Print results
            # print("Matches:", matches)
            # print("Unmatched Objects:", unmatched_objects)
            # print("Unmatched Kalman Filters:", unmatched_kalman)


            # update kalman filters
            for obj_idx, kalman_idx in matches:
                x, y, z = object_locations[obj_idx]
                kalman_id = list(tracked_objects.keys())[kalman_idx]
                tracked_objects[kalman_id].update(np.array([[x], [y], [z]]))

            # object not matched create new kalman filter
            for obj_idx in unmatched_objects:
                x, y, z = object_locations[obj_idx]
                assigned_color = np.array(colors[str(track_index)])[::-1] / 255
                new_kalman = KalmanFilter(track_index, x, y, z, object_types[obj_idx], assigned_color)
                print(f"\033[92mNew object detected. Creating Kalman Filter {new_kalman}\033[0m")
                tracked_objects[track_index] = new_kalman
                track_index += 1

            # Collect Kalman filters to remove
            to_remove = []

            for kalman in tracked_objects.values():
                if kalman.disconnect_count > DISCONNECT_THRESHOLD:
                    to_remove.append(kalman.id)

            # Now delete the Kalman filters after the iteration
            for kalman_id in to_remove:
                print(f"\033[91mRemoving Kalman Filter {tracked_objects[kalman_id]}\033[0m")
                del tracked_objects[kalman_id]

            # plot points
            for id, obj in tracked_objects.items():
                x, y, z = obj.get_location()
                if obj.disconnect_count != 0:
                    plot_point(ax2, x, y, z, obj.color, obj.object_type, id, alpha=0.25)
                else:
                    plot_point(ax2, x, y, z, obj.color, obj.object_type, id)

            for i in range(len(object_locations)):
                plot_point(ax1, object_locations[i][0], object_locations[i][1], object_locations[i][2], 'blue', object_types[i], -1)
            plot_points(ax1, ax2)
            current_frame += 1
            object_locations = []
            object_types = []

        object_type = get_object_type(label)
        x, y, z = get_location(label)
        object_locations.append((x, y, z))
        object_types.append(object_type)
