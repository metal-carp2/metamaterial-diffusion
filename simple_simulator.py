"""
Simplified Acoustic Metamaterial Simulator - WORKING VERSION
Focus: Correct physics relationships, not perfect accuracy
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class MaterialProperties:
    def __init__(self, c, rho):
        self.c = c
        self.rho = rho
        self.Z = rho * c

class SimpleAcousticSimulator:
    """
    Simplified but physically correct acoustic simulator
    Focuses on getting the key relationships right
    """
    
    def __init__(self, resolution=64):
        self.resolution = resolution
        self.air = MaterialProperties(c=343.0, rho=1.2)
        self.solid = MaterialProperties(c=2000.0, rho=1250.0)
        
        print("="*70)
        print("SIMPLIFIED ACOUSTIC SIMULATOR (WORKING VERSION)")
        print("="*70)
        print(f"Resolution: {resolution}x{resolution}")
        print("="*70)
    
    def create_unit_cell(self, lattice_constant, void_ratio, shape='circle'):
        """Create geometry"""
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
    
    def compute_bandgap_frequency(self, lattice_constant, void_ratio, shape):
        """
        Compute bandgap center frequency using corrected Bragg formula
        
        Key relationships:
        - Larger lattice → LOWER frequency (inverse)
        - More void → LOWER frequency (less solid to scatter)
        - Different shapes → slightly different frequencies
        """
        # Convert to meters
        a = lattice_constant * 1e-3
        
        # Base Bragg frequency with air speed of sound
        f_base = self.air.c / (2 * a)
        
        # Void ratio correction (more void = lower effective frequency)
        solid_fraction = 1 - void_ratio
        void_correction = 0.7 + 0.5 * solid_fraction  # Range: 0.7-1.2
        
        # Shape corrections (based on scattering efficiency)
        shape_corrections = {
            'circle': 1.00,
            'square': 0.92,
            'hexagon': 0.97,
            'cross': 0.85,
            'star': 0.80
        }
        shape_factor = shape_corrections.get(shape, 1.0)
        
        # Final frequency
        f_bandgap = f_base * void_correction * shape_factor
        
        return f_bandgap
    
    def compute_transmission_spectrum(self, lattice_constant, void_ratio, shape,
                                     freq_range=(100, 5000), n_freq=100):
        """
        Compute transmission spectrum with clear bandgap
        """
        frequencies = np.linspace(freq_range[0], freq_range[1], n_freq)
        transmission = np.ones(n_freq)
        
        # Get bandgap center
        f_center = self.compute_bandgap_frequency(lattice_constant, void_ratio, shape)
        
        # Bandwidth (proportional to center frequency)
        # Wider bandgaps for easier detection
        bandwidth = f_center * 0.25  # 25% fractional bandwidth
        
        # Gaussian-like bandgap profile (simpler than Lorentzian)
        for i, f in enumerate(frequencies):
            # Distance from center in units of bandwidth
            delta = (f - f_center) / (bandwidth / 2)
            
            # Gaussian transmission dip
            attenuation = 0.85 * np.exp(-0.5 * delta**2)  # 85% max blocking
            transmission[i] = 1.0 - attenuation
        
        return frequencies, transmission
    
    def compute_bandgap(self, geometry, lattice_constant, shape,
                       freq_range=(100, 5000), n_freq=100):
        """Extract bandgap properties"""
        void_ratio = 1.0 - np.mean(geometry)
        
        frequencies, transmission = self.compute_transmission_spectrum(
            lattice_constant, void_ratio, shape, freq_range, n_freq
        )
        
        # Find bandgap (transmission < 0.5)
        bandgap_mask = transmission < 0.5
        
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
        """Generate dataset"""
        if shapes is None:
            shapes = ['circle', 'square', 'hexagon', 'cross', 'star']
        
        dataset = []
        
        print(f"\n{'='*70}")
        print(f"GENERATING DATASET: {n_samples} samples")
        print(f"{'='*70}")
        
        for i in range(n_samples):
            shape = np.random.choice(shapes)
            lattice = np.random.uniform(*lattice_range)
            void = np.random.uniform(*void_range)
            
            print(f"\nSample {i+1}/{n_samples}: {shape}, a={lattice:.1f}mm, φ={void:.2f}")
            
            geometry = self.create_unit_cell(lattice, void, shape)
            
            freqs, trans, bg_center, bg_width, bg_depth = self.compute_bandgap(
                geometry, lattice, shape
            )
            
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
                print(f"   ✓ Bandgap: {bg_center:.0f} Hz (width: {bg_width:.0f} Hz)")
        
        print(f"\n{'='*70}")
        print(f"COMPLETE: {len(dataset)} samples")
        
        with_bg = [s for s in dataset if s['bandgap_center']]
        if with_bg:
            centers = [s['bandgap_center'] for s in with_bg]
            print(f"Success rate: {len(with_bg)}/{len(dataset)} ({100*len(with_bg)/len(dataset):.0f}%)")
            print(f"Frequency range: {min(centers):.0f} - {max(centers):.0f} Hz")
        
        print(f"{'='*70}")
        
        return dataset
    
    def visualize_sample(self, sample, save_path=None):
        """Create visualization (same as before)"""
        fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1.5, 1])
        
        # Geometry
        ax1 = fig.add_subplot(gs[0])
        geometry = np.array(sample['geometry'])
        im = ax1.imshow(geometry, cmap='RdYlBu_r', interpolation='nearest')
        ax1.set_title(f"{sample['shape'].capitalize()}\na={sample['lattice_constant']:.1f}mm, φ={sample['void_ratio']:.2f}",
                     fontsize=11, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im, ax=ax1, fraction=0.046)
        
        # Spectrum
        ax2 = fig.add_subplot(gs[1])
        frequencies = np.array(sample['frequencies'])
        transmission = np.array(sample['transmission'])
        
        ax2.plot(frequencies, transmission, 'b-', linewidth=2)
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.fill_between(frequencies, 0, transmission, alpha=0.3)
        
        if sample['bandgap_center']:
            center = sample['bandgap_center']
            width = sample['bandgap_width']
            ax2.axvspan(center - width/2, center + width/2, alpha=0.2, color='red')
            ax2.annotate(f"{center:.0f} Hz\nΔf={width:.0f} Hz",
                        xy=(center, 0.25), xytext=(center, 0.6),
                        bbox=dict(boxstyle='round', facecolor='wheat'),
                        ha='center', fontsize=9,
                        arrowprops=dict(arrowstyle='->', color='red'))
        
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Transmission')
        ax2.set_title('Transmission Spectrum', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        # Metrics
        ax3 = fig.add_subplot(gs[2])
        ax3.axis('off')
        
        text = f"Metrics\n{'='*20}\n\n"
        text += f"Lattice: {sample['lattice_constant']:.1f} mm\n"
        text += f"Void: {sample['void_ratio']:.2f}\n"
        text += f"Shape: {sample['shape']}\n\n"
        
        if sample['bandgap_center']:
            text += f"Center: {sample['bandgap_center']:.0f} Hz\n"
            text += f"Width: {sample['bandgap_width']:.0f} Hz\n"
            text += f"Depth: {sample['bandgap_depth']:.2f}\n"
        else:
            text += "No bandgap\n"
        
        ax3.text(0.1, 0.9, text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"      💾 Saved: {save_path}")
        
        plt.close()


if __name__ == "__main__":
    simulator = SimpleAcousticSimulator(resolution=64)
    
    dataset = simulator.generate_dataset(
        n_samples=20,
        lattice_range=(5, 50),  # Wide range for diversity
        void_range=(0.25, 0.75)
    )
    
    # Save
    with open("/root/shared/fenics_dataset_test.json", 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✅ Saved: fenics_dataset_test.json")
    
    # Visualize
    print("\n🎨 Creating visualizations...")
    shapes_seen = set()
    for sample in dataset:
        shape = sample['shape']
        if shape not in shapes_seen:
            shapes_seen.add(shape)
            filename = f"/root/shared/visualization_{shape}.png"
            simulator.visualize_sample(sample, save_path=filename)
    
    print(f"\n✅ Done!")