"""Utilities for validating and fixing inertia tensors in SDF files.

SDF files written during scene generation can have inertia tensors that fail
Drake's CouldBePhysicallyValid() check due to roundoff errors violating the
triangle inequality (Imin + Imed >= Imax). This module provides a safety net
that minimally fixes such violations via eigenvalue projection.
"""

import logging
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

console_logger = logging.getLogger(__name__)

# Small positive value used as minimum eigenvalue and epsilon for triangle
# inequality enforcement.
_EIGENVALUE_EPSILON = 1e-10
_TRIANGLE_RELATIVE_MARGIN = 1e-8


def _triangle_margin(sorted_eigenvalues: np.ndarray) -> float:
    """Return a scale-aware margin that survives text serialization."""
    scale = max(float(np.max(np.abs(sorted_eigenvalues))), _EIGENVALUE_EPSILON)
    return max(_EIGENVALUE_EPSILON, scale * _TRIANGLE_RELATIVE_MARGIN)


def ensure_valid_inertia(inertia_tensor: np.ndarray) -> np.ndarray:
    """Ensure an inertia tensor satisfies Drake's physical validity checks.

    Fixes two issues via eigenvalue projection:
    1. Negative eigenvalues (clamped to small positive value).
    2. Triangle inequality violations: for sorted eigenvalues
       [e1 <= e2 <= e3], need e1 + e2 >= e3.

    Returns the original tensor if already valid (no-op).

    Args:
        inertia_tensor: 3x3 symmetric inertia matrix.

    Returns:
        Fixed 3x3 symmetric inertia matrix, or the original if valid.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)

    # Check if any fix is needed.
    needs_fix = False

    if np.any(eigenvalues < _EIGENVALUE_EPSILON):
        needs_fix = True

    # Sort eigenvalues for triangle inequality check.
    sorted_eigs = np.sort(eigenvalues)
    triangle_margin = _triangle_margin(sorted_eigs)
    if sorted_eigs[0] + sorted_eigs[1] < sorted_eigs[2] + triangle_margin:
        needs_fix = True

    if not needs_fix:
        return inertia_tensor

    # Clamp negative or near-zero eigenvalues.
    fixed_eigenvalues = np.maximum(eigenvalues, _EIGENVALUE_EPSILON)

    # Check and fix triangle inequality on sorted values.
    sorted_indices = np.argsort(fixed_eigenvalues)
    sorted_eigs = fixed_eigenvalues[sorted_indices]

    triangle_margin = _triangle_margin(sorted_eigs)
    if sorted_eigs[0] + sorted_eigs[1] < sorted_eigs[2] + triangle_margin:
        # Scale up the two smaller eigenvalues minimally so that
        # e1 + e2 = e3 + epsilon.
        deficit = sorted_eigs[2] + triangle_margin - (sorted_eigs[0] + sorted_eigs[1])
        # The equal split is the minimum-L2 projection onto the half-space and
        # avoids pushing only the middle eigenvalue above the old maximum (which
        # would merely move the violation to a different sorted index).
        sorted_eigs[0] += deficit / 2
        sorted_eigs[1] += deficit / 2

        # Write back to unsorted order.
        fixed_eigenvalues[sorted_indices] = sorted_eigs

    # Reconstruct: I_fixed = V @ diag(fixed_eigenvalues) @ V^T.
    fixed_tensor = eigenvectors @ np.diag(fixed_eigenvalues) @ eigenvectors.T

    # Ensure symmetry (eliminate floating point asymmetry).
    fixed_tensor = (fixed_tensor + fixed_tensor.T) / 2

    return fixed_tensor


def fix_sdf_file_inertia(sdf_path: Path) -> bool:
    """Parse an SDF file and fix any inertia tensors that violate Drake's
    physical validity checks.

    Preserves mass, center-of-mass pose, and all other SDF content. Only
    modifies inertia tensor components if they fail the triangle inequality
    or have negative eigenvalues.

    Args:
        sdf_path: Path to SDF file to fix in-place.

    Returns:
        True if any inertia tensors were modified.
    """
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    any_modified = False

    for inertia_elem in root.iter("inertia"):
        # Extract the 6 unique components of the symmetric 3x3 tensor.
        ixx = float(inertia_elem.findtext("ixx", "0"))
        iyy = float(inertia_elem.findtext("iyy", "0"))
        izz = float(inertia_elem.findtext("izz", "0"))
        ixy = float(inertia_elem.findtext("ixy", "0"))
        ixz = float(inertia_elem.findtext("ixz", "0"))
        iyz = float(inertia_elem.findtext("iyz", "0"))

        tensor = np.array(
            [
                [ixx, ixy, ixz],
                [ixy, iyy, iyz],
                [ixz, iyz, izz],
            ]
        )

        fixed_tensor = ensure_valid_inertia(tensor)

        if np.array_equal(fixed_tensor, tensor):
            console_logger.debug("Inertia tensor in '%s' already valid", sdf_path.name)
            continue

        console_logger.warning(
            "Fixing inertia tensor in '%s' to satisfy triangle inequality",
            sdf_path.name,
        )

        # Write back the fixed components.
        # Twelve decimal digits retain the projected safety margin after SDF
        # round-trip parsing. Six digits can round a valid boundary tensor back
        # into a Drake triangle-inequality violation.
        inertia_elem.find("ixx").text = f"{fixed_tensor[0, 0]:.12e}"
        inertia_elem.find("iyy").text = f"{fixed_tensor[1, 1]:.12e}"
        inertia_elem.find("izz").text = f"{fixed_tensor[2, 2]:.12e}"
        inertia_elem.find("ixy").text = f"{fixed_tensor[0, 1]:.12e}"
        inertia_elem.find("ixz").text = f"{fixed_tensor[0, 2]:.12e}"
        inertia_elem.find("iyz").text = f"{fixed_tensor[1, 2]:.12e}"

        any_modified = True

    if any_modified:
        ET.indent(root, space="  ", level=0)
        tree.write(sdf_path, encoding="utf-8", xml_declaration=True)

    return any_modified
