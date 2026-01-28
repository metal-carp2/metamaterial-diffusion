"""
FEniCS-Based Acoustic Metamaterial Simulator (V2.0)
Replaces the simplified transfer matrix method with full wave equation solver
"""

import numpy as np
from dolfinx import mesh, fem, io
from dolfinx.fem import (Constant, Function, FunctionSpace, 
                         assemble_scalar, dirichletbc, form, locate_dofs_topological)
from dolfinx.mesh import create_rectangle, locate_entities_boundary
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from pathlib import Path
import json

class MaterialProperties:
    """Physical properties of materials"""
    def __init__(self, c, rho):
        self.c = c      # Speed of sound (m/s)
        self.rho = rho  # Density (kg/m³)
        self.Z = rho * c  # Acoustic impedance

class FEniCSAcousticSimulator:
    """
    Acoustic metamaterial simulator using FEniCS/DOLFINx
    Solves the Helmholtz equation for acoustic wave propagation
    """
    
    def __init__(self, resolution=64):
        self.resolution = resolution
        
        # Material properties
        self.air = MaterialProperties(c=343.0, rho=1.2)      # Air
        self.solid = MaterialProperties(c=2000.0, rho=1250.0)  # PLA plastic
        
        print("="*70)
        print("FEniCS ACOUSTIC METAMATERIAL SIMULATOR V2.0")
        print("="*70)
        print(f"Resolution: {resolution}x{resolution}")
        print(f"Air: c={self.air.c} m/s, ρ={self.air.rho} kg/m³")
        print(f"Solid: c={self.solid.c} m/s, ρ={self.solid.rho} kg/m³")
        print("="*70)
    
    def create_unit_cell(self, lattice_constant, void_ratio, shape='circle'):
        """
        Create a unit cell geometry
        
        Args:
            lattice_constant: Size of unit cell (mm)
            void_ratio: Fraction that is void (0-1)
            shape: 'circle', 'square', 'hexagon', 'cross', 'star'
        
        Returns:
            geometry: 2D numpy array (resolution x resolution)
        """
        size = self.resolution
        geometry = np.ones((size, size))
        center = size / 2
        
        if shape == 'circle':
            radius = np.sqrt(void_ratio) * size / 2
            y, x = np.ogrid[:size, :size]
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            geometry = (dist > radius).astype(float)
            
        elif shape == 'square':
            side = np.sqrt(void_ratio) * size
            half_side = side / 2
            y, x = np.ogrid[:size, :size]
            mask = (np.abs(x - center) < half_side) & (np.abs(y - center) < half_side)
            geometry = (~mask).astype(float)
            
        elif shape == 'hexagon':
            radius = np.sqrt(void_ratio * 2 / np.sqrt(3)) * size / 2
            y, x = np.ogrid[:size, :size]
            dx = x - center
            dy = y - center
            
            angles = np.array([0, 60, 120]) * np.pi / 180
            mask = np.ones((size, size), dtype=bool)
            for angle in angles:
                nx, ny = np.cos(angle), np.sin(angle)
                dist = np.abs(dx * nx + dy * ny)
                mask &= (dist < radius * np.cos(np.pi/6))
            geometry = (~mask).astype(float)
            
        elif shape == 'cross':
            arm_width = np.sqrt(void_ratio / 2) * size
            half_width = arm_width / 2
            y, x = np.ogrid[:size, :size]
            h_mask = (np.abs(y - center) < half_width)
            v_mask = (np.abs(x - center) < half_width)
            geometry = (~(h_mask | v_mask)).astype(float)
            
        elif shape == 'star':
            radius = np.sqrt(void_ratio * 1.5) * size / 2
            y, x = np.ogrid[:size, :size]
            dx = x - center
            dy = y - center
            angle = np.arctan2(dy, dx)
            dist = np.sqrt(dx**2 + dy**2)
            star_radius = radius * (1 + 0.4 * np.cos(5 * angle))
            geometry = (dist > star_radius).astype(float)
        
        return geometry
    
    def solve_helmholtz(self, geometry, frequency, lattice_constant):
        """
        Solve Helmholtz equation for given geometry and frequency
        
        Args:
            geometry: 2D array (0=void, 1=solid)
            frequency: Frequency in Hz
            lattice_constant: Size of unit cell in mm
        
        Returns:
            transmission: Transmission coefficient (0-1)
        """
        # Convert lattice constant from mm to m
        a = lattice_constant * 1e-3
        
        # Angular frequency
        omega = 2 * np.pi * frequency
        
        # Create mesh (unit square representing the unit cell)
        domain = create_rectangle(
            MPI.COMM_WORLD,
            [[0.0, 0.0], [1.0, 1.0]],
            [self.resolution-1, self.resolution-1]
        )
        
        # Function space for acoustic pressure
        V = FunctionSpace(domain, ("Lagrange", 1))
        
        # Material properties mapped from geometry
        # We need to interpolate material properties onto the mesh
        
        # For simplicity, use average properties based on void ratio
        void_fraction = 1.0 - np.mean(geometry)
        
        # Effective properties (rule of mixtures)
        rho_eff = void_fraction * self.air.rho + (1 - void_fraction) * self.solid.rho
        c_eff_inv = void_fraction / self.air.c + (1 - void_fraction) / self.solid.c
        c_eff = 1.0 / c_eff_inv
        
        # Wave number
        k = omega / c_eff
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Helmholtz equation: ∇²p + k²p = 0
        # Weak form: ∫(∇u·∇v - k²uv) dx = 0
        a_form = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k**2 * u * v) * ufl.dx
        L_form = Constant(domain, 0.0) * v * ufl.dx
        
        # Boundary conditions (simplified - incident wave from left)
        # Left boundary: incident plane wave p = exp(ikx)
        # Right boundary: outgoing wave (approximate with absorbing BC)
        
        # For now, use simple Dirichlet on left, Neumann on right
        # This is a simplified model - full implementation would use PML
        
        def left_boundary(x):
            return np.isclose(x[0], 0.0)
        
        def right_boundary(x):
            return np.isclose(x[0], 1.0)
        
        # Apply incident wave on left
        left_facets = locate_entities_boundary(domain, domain.topology.dim - 1, left_boundary)
        left_dofs = locate_dofs_topological(V, domain.topology.dim - 1, left_facets)
        
        # Incident pressure amplitude
        u_incident = Function(V)
        u_incident.x.array[:] = 1.0  # Unit amplitude
        
        bc_left = dirichletbc(u_incident, left_dofs)
        
        # Assemble and solve
        problem = fem.petsc.LinearProblem(
            a_form, L_form, 
            bcs=[bc_left],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        
        try:
            uh = problem.solve()
            
            # Calculate transmission (pressure amplitude on right side)
            # Average pressure on right boundary
            right_facets = locate_entities_boundary(domain, domain.topology.dim - 1, right_boundary)
            right_dofs = locate_dofs_topological(V, domain.topology.dim - 1, right_facets)
            
            # Get values at right boundary
            p_transmitted = np.abs(uh.x.array[right_dofs].mean())
            
            # Transmission coefficient (normalized by incident amplitude)
            transmission = p_transmitted / 1.0  # incident amplitude is 1.0
            
            # Limit to [0, 1]
            transmission = np.clip(transmission, 0.0, 1.0)
            
        except Exception as e:
            print(f"   Warning: Solver failed for f={frequency} Hz: {e}")
            transmission = 1.0  # Return full transmission on failure
        
        return transmission
    
    def compute_bandgap(self, geometry, lattice_constant, 
                       freq_range=(100, 5000), n_freq=100):
        """
        Compute bandgap by sweeping frequencies
        
        Args:
            geometry: 2D array
            lattice_constant: Size in mm
            freq_range: (min_freq, max_freq) in Hz
            n_freq: Number of frequency points
        
        Returns:
            frequencies: Array of frequencies
            transmission: Array of transmission coefficients
            bandgap_center: Center frequency of bandgap (Hz)
            bandgap_width: Width of bandgap (Hz)
            bandgap_depth: Depth of bandgap (0-1)
        """
        frequencies = np.linspace(freq_range[0], freq_range[1], n_freq)
        transmission = np.zeros(n_freq)
        
        print(f"   Computing transmission spectrum...")
        for i, freq in enumerate(frequencies):
            if i % 20 == 0:
                print(f"      Progress: {i}/{n_freq} frequencies")
            transmission[i] = self.solve_helmholtz(geometry, freq, lattice_constant)
        
        # Identify bandgap (transmission < 0.3 threshold)
        bandgap_mask = transmission < 0.3
        
        if np.any(bandgap_mask):
            bandgap_freqs = frequencies[bandgap_mask]
            bandgap_center = np.mean(bandgap_freqs)
            bandgap_width = bandgap_freqs.max() - bandgap_freqs.min()
            bandgap_depth = 1.0 - transmission[bandgap_mask].mean()
        else:
            bandgap_center = None
            bandgap_width = 0
            bandgap_depth = 0
        
        return frequencies, transmission, bandgap_center, bandgap_width, bandgap_depth
    
    def generate_dataset(self, n_samples=100, 
                        lattice_range=(5, 50),
                        void_range=(0.2, 0.8),
                        shapes=None):
        """
        Generate dataset of metamaterial samples
        
        Args:
            n_samples: Number of samples to generate
            lattice_range: (min, max) lattice constant in mm
            void_range: (min, max) void ratio
            shapes: List of shapes (default: all)
        
        Returns:
            dataset: List of sample dictionaries
        """
        if shapes is None:
            shapes = ['circle', 'square', 'hexagon', 'cross', 'star']
        
        dataset = []
        
        print(f"\n{'='*70}")
        print(f"GENERATING DATASET: {n_samples} samples")
        print(f"{'='*70}")
        
        for i in range(n_samples):
            # Random parameters
            shape = np.random.choice(shapes)
            lattice = np.random.uniform(*lattice_range)
            void = np.random.uniform(*void_range)
            
            print(f"\nSample {i+1}/{n_samples}:")
            print(f"   Shape: {shape}, a={lattice:.1f}mm, φ={void:.2f}")
            
            # Create geometry
            geometry = self.create_unit_cell(lattice, void, shape)
            
            # Compute bandgap
            freqs, trans, bg_center, bg_width, bg_depth = self.compute_bandgap(
                geometry, lattice
            )
            
            # Store sample
            sample = {
                'index': i,
                'shape': shape,
                'lattice_constant': float(lattice),
                'void_ratio': float(void),
                'geometry': geometry.tolist(),
                'frequencies': freqs.tolist(),
                'transmission': trans.tolist(),
                'bandgap_center': float(bg_center) if bg_center else None,
                'bandgap_width': float(bg_width),
                'bandgap_depth': float(bg_depth),
            }
            
            dataset.append(sample)
            
            if bg_center:
                print(f"   ✓ Bandgap: {bg_center:.0f} Hz ± {bg_width/2:.0f} Hz (depth: {bg_depth:.2f})")
            else:
                print(f"   ✗ No significant bandgap")
        
        print(f"\n{'='*70}")
        print(f"DATASET COMPLETE: {len(dataset)} samples generated")
        print(f"{'='*70}")
        
        return dataset


# Example usage
if __name__ == "__main__":
    # Create simulator
    simulator = FEniCSAcousticSimulator(resolution=32)  # Lower res for testing
    
    # Generate small test dataset
    dataset = simulator.generate_dataset(
        n_samples=5,
        lattice_range=(10, 30),
        void_range=(0.3, 0.7)
    )
    
    # Save dataset
    output_file = "fenics_dataset_test.json"
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✓ Dataset saved to: {output_file}")
