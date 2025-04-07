#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from svgpathtools import svg2paths


def find_t_for_length(path, target_length, tol=1e-6):
    """
    invert the arc-length mapping via binary search
    """
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if path.length(0, mid) < target_length:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def process_svg_file(filepath, num_samples=100):
    """
    process an svg file that contains multiple paths (repeats of the same word).
    for each path:
      - uniformly sample 'num_samples' points along the arc length
      - convert the absolute coordinates into relative coordinates such that:
          point[i] = point[i] - point[0] for i >= 0
      - regularize the coordinates to a range close to [0,1] while preserving aspect ratio
    returns:
      a list of numpy arrays of shape (num_samples, 2), one for each path
    """
    try:
        # prune out empty elements (with no 'd' key)
        paths, attrs = svg2paths(filepath)
        valid_indices = [i for (i, attr) in enumerate(attrs) if 'd' in attr]
        paths = [paths[i] for i in valid_indices]
        attrs = [attrs[i] for i in valid_indices]
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

    if not paths:
        print(f"No paths found in {filepath}. Skipping.")
        return None

    all_points = []
    for path in paths:
        total_length = path.length()
        distances = np.linspace(0, total_length, num_samples)
        points = []
        for d in distances:
            t = find_t_for_length(path, d)
            pt = path.point(t)
            # flip y to reverse water-reflection
            points.append([pt.real, -pt.imag])
        # regularize point range while preserving aspect ratio
        points = np.array(points)
        points /= 100
        # compute relative positions to the first point
        points = points - points[0]
        all_points.append(points)
    return all_points


def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    # dictionary to store the results:
    # key: word (derived from file name, without extension)
    # value: numpy array of shape (n_repeats, num_samples, 2)
    data = {}
    num_samples = 75

    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith('.svg'):
            filepath = os.path.join(folder, filename)
            # assume the filename (without extension) is the word label
            word_label = os.path.splitext(filename)[0]
            delta_list = process_svg_file(filepath, num_samples=num_samples)
            if delta_list is None or len(delta_list) == 0:
                continue
            # convert the list of arrays to a single numpy array of shape (n_repeats, num_samples, 2)
            data[word_label] = np.array(delta_list)
            print(
                f"Processed '{filename}' for word '{word_label}': {data[word_label].shape}"
            )

    # save the data to an NPZ file
    output_filename = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "all_words.npz")
    np.savez(output_filename, **data)
    print(f"Data saved to '{output_filename}'.")

    # optionally, plot one example (first repetition) per word
    for word, samples in data.items():
        plt.figure(figsize=(5, 5))
        # reconstruct the absolute positions from the delta representation using cumulative sum
        sample = samples[0]  # choose the first repetition for plotting
        xs = sample[:, 0]
        ys = sample[:, 1]
        plt.plot(xs, ys, marker='o', markersize=2, linestyle='-')
        plt.title(f"Word: {word}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
