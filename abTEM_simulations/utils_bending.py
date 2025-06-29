import numpy as np
import matplotlib.pyplot as plt

def load_prz_path(path):
    sig = np.load(path, allow_pickle=True)
    dataset = sig['data']
    return dataset


def make_monolayer(Width, Height, cif_path, remove_na=False):
    from ase.io import read
    from ase import Atoms
    cif = read(cif_path)
    monolayer = cif * (Width, Height, 1)

    if remove_na:
        monolayer = Atoms([atom for atom in monolayer if atom.symbol != 'Na'])

    return monolayer

def apply_wrinkle_with_strain(coords, wavelength, amplitude, axis="y"): #
    if wavelength == 0 :
        return coords
    if wavelength <= 0:
        raise ValueError("Wavelength must be a positive value.")
    if amplitude <= 0:
        raise ValueError("Amplitude must be a positive value.")
    
    if coords.shape[1] != 3:
        raise ValueError("Input coordinates must have shape Nx3.")
    
    new_coords = np.copy(coords)
    period = wavelength / 2
    strain_factor = 1 + amplitude / period  # Simplified strain factor
    
    if axis == "y":
        # Apply wrinkle in the y-direction and adjust x-coordinates for strain
        y_coords = coords[:, 1]
        for i, y in enumerate(y_coords):
            new_coords[i, 2] += amplitude * np.sin(2 * np.pi * y / wavelength)
            new_coords[i, 0] *= strain_factor  # Apply strain in the x-direction
    elif axis == "x":
        # Apply wrinkle in the x-direction and adjust y-coordinates for strain
        x_coords = coords[:, 0]
        for i, x in enumerate(x_coords):
            new_coords[i, 2] += amplitude * np.sin(2 * np.pi * x / wavelength)
            new_coords[i, 1] *= strain_factor  # Apply strain in the y-direction
    elif axis == "z":
        # Apply wrinkle in the z-direction and adjust x and y-coordinates for strain
        z_coords = coords[:, 2]
        for i, z in enumerate(z_coords):
            new_coords[i, 1] += amplitude * np.sin(2 * np.pi * z / wavelength)
            new_coords[i, 0] *= strain_factor  # Apply strain in the x-direction
            new_coords[i, 1] *= strain_factor  # Apply strain in the y-direction
            
    else:
        raise ValueError("Invalid axis. Please choose 'x' or 'y'.")
    
    return new_coords

# Example usage
# coords = PHI_monolayer.positions
# wavelength = 50.0  # Wavelength of the wrinkle
# amplitude = 10.0  # Amplitude of the wrinkle

# wrinkled_coords = apply_wrinkle_with_strain(coords, wavelength, amplitude, axis="y")

# PHI_wrinkled = PHI_monolayer.copy()
# PHI_wrinkled.positions = wrinkled_coords
# write("./tmp/test_wrinkle_strain.xyz",PHI_wrinkled)

import numpy as np

def apply_wrinkle(coords, wavelength, amplitude, axis="y"):
    if wavelength <= 0:
        raise ValueError("Wavelength must be a positive value.")
    if amplitude <= 0:
        raise ValueError("Amplitude must be a positive value.")
    
    if coords.shape[1] != 3:
        raise ValueError("Input coordinates must have shape Nx3.")
    
    new_coords = np.copy(coords)
    
    if axis == "y":
        # Apply wrinkle in the y-direction
        new_coords[:, 2] += amplitude * np.sin(2 * np.pi * coords[:, 1] / wavelength)
    elif axis == "x":
        # Apply wrinkle in the x-direction
        new_coords[:, 2] += amplitude * np.sin(2 * np.pi * coords[:, 0] / wavelength)
    elif axis == "z":
        # Apply wrinkle in the z-direction
        new_coords[:, 1] += amplitude * np.sin(2 * np.pi * coords[:, 2] / wavelength)
    else:
        raise ValueError("Invalid axis. Please choose 'x' or 'y'.")
    
    return new_coords

# # Example usage
# #coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])  # Replace with actual coordinates
# coords = PHI_monolayer.positions
# wavelength = 50.0  # Wavelength of the wrinkle
# amplitude = 10.0  # Amplitude of the wrinkle
# axis = "y"  # Wrinkle along the y-axis

# wrinkled_coords = apply_wrinkle(coords, wavelength, amplitude, axis)
# print(wrinkled_coords)
# PHI_wrinkled = PHI_monolayer.copy()
# PHI_wrinkled.positions = wrinkled_coords
# write("./tmp/test_wrinkle.xyz",PHI_wrinkled)

import numpy as np

def create_2d_strain_matrix(epsilon_xx_percent, epsilon_yy_percent, epsilon_xy_percent):
    """
    Create a 3x3 strain matrix for 2D strain, with the last row as [0, 0, 1].
    
    Parameters:
    epsilon_xx_percent (float): Strain in the xx direction in percent.
    epsilon_yy_percent (float): Strain in the yy direction in percent.
    epsilon_xy_percent (float): Shear strain in the xy plane in percent.
    
    Returns:
    np.ndarray: A 3x3 strain matrix.
    """
    
    # Convert percent strains to decimal form
    epsilon_xx = epsilon_xx_percent / 100.0
    epsilon_yy = epsilon_yy_percent / 100.0
    epsilon_xy = epsilon_xy_percent / 100.0

    # Construct the strain tensor
    strain_matrix = np.array([[1 + epsilon_xx, epsilon_xy, 0],
                              [epsilon_xy, 1 + epsilon_yy, 0],
                              [0, 0, 1]])

    return strain_matrix


import numpy as np

def introduce_stacking_faults(positions, fault_plane='xy', shift_vector=None, fault_start_idx=0, fault_end_idx=None):
    """
    Introduces a stacking fault in the given atomistic monolayer position matrix.
    
    Parameters:
    - positions: Nx3 numpy array of atomic positions
    - fault_plane: The plane in which to introduce the fault ('xy', 'yz', or 'zx')
    - shift_vector: The vector by which to shift the atoms in the fault region
    - fault_start_idx: The starting index of the fault region
    - fault_end_idx: The ending index of the fault region (if None, it defaults to the end of the array)
    
    Returns:
    - new_positions: Nx3 numpy array of atomic positions with the stacking fault introduced
    """
    if fault_end_idx is None:
        fault_end_idx = len(positions)
    
    new_positions = np.copy(positions)
    
    # Define default shift vectors if none are provided
    if shift_vector is None:
        if fault_plane == 'xy':
            shift_vector = np.array([0.5, 0.5, 0])
        elif fault_plane == 'yz':
            shift_vector = np.array([0, 0.5, 0.5])
        elif fault_plane == 'zx':
            shift_vector = np.array([0.5, 0, 0.5])
    
    # Apply the shift vector to the specified region
    new_positions[fault_start_idx:fault_end_idx, :] += shift_vector
    
    return new_positions

import numpy as np

def affine_matrix_from_strains(epsilon_xx_percent, epsilon_yy_percent, epsilon_xy_percent, epsilon_yx_percent):
    # Convert percent strains to decimal form
    epsilon_xx = epsilon_xx_percent / 100.0
    epsilon_yy = epsilon_yy_percent / 100.0
    epsilon_xy = epsilon_xy_percent / 100.0
    epsilon_yx = epsilon_yx_percent / 100.0

    # Construct the Green-Lagrange strain tensor E
    E = np.array([[epsilon_xx, epsilon_xy / 2, 0],
                  [epsilon_yx / 2, epsilon_yy, 0],
                  [0, 0, 0]])

    # Reconstruct the deformation gradient tensor F
    F = np.eye(3) + E

    # The upper-left 3x3 submatrix of F is the affine matrix
    affine_matrix = F[:3, :3]

    return affine_matrix

# # Example usage
# epsilon_xx_percent = 2  # 2% strain in the x-direction
# epsilon_yy_percent = 3  # 3% strain in the y-direction
# epsilon_xy_percent = 5  # 5% shear strain (xy)
# epsilon_yx_percent = 6  # 6% shear strain (yx)

# affine_matrix = affine_matrix_from_strains(5,5,5,5)
# print("Affine Transformation Matrix:")
# print(affine_matrix)

import numpy as np
from ase import Atoms

def bend_sheet_into_half_cylinder(coords, width, axis="y", z_constant=0): #The axis are reverse
    if width <= 0:
        raise ValueError("Width must be a positive value.")
    
    if coords.shape[1] != 3:
        raise ValueError("Input coordinates must have shape Nx3.")
    
    M, _ = coords.shape
    radius = width / np.pi
    new_coords = np.zeros_like(coords)
    
    coords[:, 2] += z_constant  # Add z_constant to all z values

    # Center the coordinates around the axis of bending
    if axis == "y":
        y_center = (np.max(coords[:, 1]) + np.min(coords[:, 1])) / 2
        coords[:, 1] -= y_center  # Center y-coordinates
        theta = np.pi * coords[:, 1] / width
        new_coords[:, 0] = coords[:, 0]  # x'
        new_coords[:, 1] = radius * np.sin(theta)  # y'
        new_coords[:, 2] = radius * np.cos(theta) - radius + coords[:, 2]  # z'
        new_coords[:, 1] += y_center  # Restore y-coordinates
    elif axis == "x":
        x_center = (np.max(coords[:, 0]) + np.min(coords[:, 0])) / 2
        coords[:, 0] -= x_center  # Center x-coordinates
        theta = np.pi * coords[:, 0] / width
        new_coords[:, 0] = radius * np.sin(theta)  # x'
        new_coords[:, 1] = coords[:, 1]  # y'
        new_coords[:, 2] = radius * np.cos(theta) - radius + coords[:, 2]  # z'
        new_coords[:, 0] += x_center  # Restore x-coordinates
    else:
        raise ValueError("Invalid axis. Please choose 'x' or 'y'.")
    
    # Calculate the rotation needed to align the bend center with the z-axis
    if axis == "y":
        rotation_angle = -np.pi / 2
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
            [0, np.sin(rotation_angle), np.cos(rotation_angle)]
        ])
    elif axis == "x":
        rotation_angle = np.pi / 2
        rotation_matrix = np.array([
            [np.cos(rotation_angle), 0, np.sin(rotation_angle)],
            [0, 1, 0],
            [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]
        ])
    
    # Apply rotation
    new_coords = new_coords @ rotation_matrix.T

    return new_coords

import numpy as np

def bend_sheet_into_cylinder(coords, bend, axis="y", z_constant=0):  # The axes are reversed!
    if bend < 0 or bend > 2 * np.pi:
        raise ValueError("Bend must be between 0 and 2*pi.")
    
    if coords.shape[1] != 3:
        raise ValueError("Input coordinates must have shape Nx3.")
    
    new_coords = np.zeros_like(coords)
    
    # Add z_constant to all z values
    coords[:, 2] += z_constant

    # Center the coordinates around the axis of bending
    if axis == "x":
        y_range = np.max(coords[:, 1]) - np.min(coords[:, 1])
        radius = y_range / bend if bend != 0 else np.inf
        y_center = (np.max(coords[:, 1]) + np.min(coords[:, 1])) / 2
        coords[:, 1] -= y_center  # Center y-coordinates
        theta = bend * coords[:, 1] / y_range
        new_coords[:, 0] = coords[:, 0]  # x'
        new_coords[:, 1] = radius * np.sin(theta)  # y'
        new_coords[:, 2] = radius * (1 - np.cos(theta)) + coords[:, 2]  # z'
        new_coords[:, 1] += y_center  # Restore y-coordinates
    elif axis == "y":
        x_range = np.max(coords[:, 0]) - np.min(coords[:, 0])
        radius = x_range / bend if bend != 0 else np.inf
        x_center = (np.max(coords[:, 0]) + np.min(coords[:, 0])) / 2
        coords[:, 0] -= x_center  # Center x-coordinates
        theta = bend * coords[:, 0] / x_range
        new_coords[:, 0] = radius * np.sin(theta)  # x'
        new_coords[:, 1] = coords[:, 1]  # y'
        new_coords[:, 2] = radius * (1 - np.cos(theta)) + coords[:, 2]  # z'
        new_coords[:, 0] += x_center  # Restore x-coordinates
    else:
        raise ValueError("Invalid axis. Please choose 'x' or 'y'.")
    
    return new_coords





from ase import Atoms
import numpy as np
from ase.io import read

def build_wrinkled_multilayer(
    monolayer,
    num_layers,
    layer_spacing,
    amplitude,
    wavelength,
    axis="y",
    rotation=0,
    random_rotation=False,
    random_wave=False,
    random_amplitude=False,
    seed=42
):
    from utils_bending import apply_wrinkle_with_strain  # Ensure correct import if placed externally

    np.random.seed(seed)
    param_choice = np.random.randint(0, 3, size=num_layers)
    rand_rotation_angs = np.random.randint(0, 9, size=num_layers)

    multilayer = monolayer.copy()
    if wavelength > 0:
        coords = multilayer.positions
        multilayer.positions = apply_wrinkle_with_strain(coords, wavelength, amplitude, axis=axis)

    for layer in range(1, num_layers):
        next_layer = monolayer.copy()
        next_layer.positions[:, 2] += layer_spacing * layer

        # Rotation
        if random_rotation:
            next_layer.rotate('z', rand_rotation_angs[layer], center="COM")
        elif rotation and layer > num_layers // 2:
            next_layer.rotate('z', rotation, center="COM")

        # Wave randomness
        current_wavelength = wavelength
        current_amplitude = amplitude

        if random_wave:
            if param_choice[layer] == 0:
                current_amplitude *= 1.0 if not random_amplitude else 0.9
            elif param_choice[layer] == 1:
                current_amplitude *= 1.0 if not random_amplitude else 1.0
            elif param_choice[layer] == 2:
                current_amplitude *= 1.0 if not random_amplitude else 1.1

        coords = next_layer.positions
        deformed = apply_wrinkle_with_strain(coords, current_wavelength, current_amplitude, axis=axis)
        next_layer.positions = deformed

        multilayer += next_layer

    return multilayer
