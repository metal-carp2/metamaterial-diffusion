"""
Resonant Cavity Acoustic Metamaterial Simulator
Uses cavity resonance physics instead of Bragg scattering

Key difference from V1.0:
- V1.0: Bragg scattering (f = c/2a) - periodic structure interference
- V2.0: Resonant cavity (f = c/L) - void acts as Helmholtz resonator

Physics basis: Each void is an acoustic resonator with resonant frequency
determined by cavity dimensions and neck geometry.
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

class ResonantCavitySimulator:
    """
    Acoustic simulator based on Helmholtz resonator physics
    Fundamentally different from Bragg scattering approach
    """
    
    def __init__(self, resolution=64):
        self.resolution = resolution
        self.air = MaterialProperties(c=343.0, rho=1.2)
        self.solid = MaterialProperties(c=2000.0, rho=1250.0)
        
        print("="*70)
        print("RESONANT CAVITY ACOUSTIC SIMULATOR (V2.0)")
        print("="*70)
        print("Physics: Helmholtz resonator model")
        print("Different from V1.0 Bragg scattering")
        print(f"Resolution: {resolution}x{resolution}")
        print("="*70)
    
    def create_unit_cell(self, lattice_constant, void_ratio, shape='circle'):
        """Create geometry - same as before"""
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
    
    def compute_cavity_dimensions(self, lattice_constant, void_ratio, shape):
        """
        Compute effective cavity dimensions for Helmholtz resonator
        
        Physics: Cavity resonance depends on volume and neck geometry
        f = (c/(2π)) * sqrt(A/(V*L))
        where A = neck area, V = cavity volume, L = neck length
        """
        # Cavity volume (proportional to void ratio and lattice size)
        a = lattice_constant * 1e-3  # Convert to meters
        V_cavity = void_ratio * (a ** 2) * a  # Volume in m³
        
        # Effective neck dimensions depend on shape
        # Different shapes have different opening characteristics
        shape_neck_factors = {
            'circle': 1.0,      # Smooth, uniform opening
            'hexagon': 0.95,    # Slightly restricted
            'square': 0.85,     # Corners create turbulence
            'cross': 0.70,      # Multiple narrow necks
            'star': 0.60        # Very restricted openings
        }
        neck_factor = shape_neck_factors.get(shape, 1.0)
        
        # Effective neck area
        A_neck = void_ratio * (a ** 2) * neck_factor
        
        # Effective neck length (wall thickness)
        L_neck = (1 - void_ratio) * a
        
        return V_cavity, A_neck, L_neck
    
    def compute_resonant_frequency(self, lattice_constant, void_ratio, shape):
        """
        Compute resonant frequency using Helmholtz resonator formula
        
        This is DIFFERENT from Bragg scattering (V1.0)!
        """
        V_cavity, A_neck, L_neck = self.compute_cavity_dimensions(
            lattice_constant, void_ratio, shape
        )
        
        # Helmholtz resonator frequency
        # f = (c/(2π)) * sqrt(A/(V*L_eff))
        # where L_eff = L + 0.85*sqrt(A) (end correction)
        
        L_eff = L_neck + 0.85 * np.sqrt(A_neck)
        
        if V_cavity > 0 and L_eff > 0:
            f_resonant = (self.air.c / (2 * np.pi)) * np.sqrt(A_neck / (V_cavity * L_eff))
        else:
            f_resonant = 1000  # Fallback
        
        # Add small randomness for diversity
        noise = np.random.uniform(0.95, 1.05)
        f_resonant *= noise
        
        return f_resonant
    
    def compute_transmission_spectrum(self, lattice_constant, void_ratio, shape,
                                     freq_range=(100, 5000), n_freq=100):
        """Compute transmission spectrum using resonant cavity model"""
        frequencies = np.linspace(freq_range[0], freq_range[1], n_freq)
        
        # Resonant frequency from cavity physics
        f_resonant = self.compute_resonant_frequency(lattice_constant, void_ratio, shape)
        
        # Quality factor (depends on losses in cavity)
        # More void = lower Q (more losses)
        Q = 2.0 + 6.0 * (1 - void_ratio)  # Q = 2-8
        
        bandwidth = f_resonant / Q
        
        transmission = np.ones(n_freq)
        for i, f in enumerate(frequencies):
            # Resonant absorption
            delta = (f - f_resonant) / (bandwidth / 2)
            lorentzian = 1.0 / (1.0 + delta**2)
            
            # Absorption depth (depends on cavity coupling)
            max_absorption = 0.75 + 0.20 * (1 - void_ratio)
            
            transmission[i] = 1.0 - max_absorption * lorentzian
        
        return frequencies, transmission
    
    def compute_bandgap(self, geometry, lattice_constant, shape,
                       freq_range=(100, 5000), n_freq=100):
        """Extract bandgap properties"""
        void_ratio = 1.0 - np.mean(geometry)
        
        frequencies, transmission = self.compute_transmission_spectrum(
            lattice_constant, void_ratio, shape, freq_range, n_freq
        )
        
        # Detect bandgap
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
                        shapes=None,
                        ensure_equal_shapes=True):
        """Generate dataset with resonant cavity physics"""
        if shapes is None:
            shapes = ['circle', 'square', 'hexagon', 'cross', 'star']
        
        dataset = []
        
        print(f"\n{'='*70}")
        print(f"GENERATING DATASET: {n_samples} samples")
        print(f"Physics: Resonant cavity (Helmholtz resonator)")
        print(f"{'='*70}")
        
        # Equal shape distribution
        if ensure_equal_shapes:
            samples_per_shape = n_samples // len(shapes)
            shape_sequence = []
            for shape in shapes:
                shape_sequence.extend([shape] * samples_per_shape)
            remainder = n_samples - len(shape_sequence)
            shape_sequence.extend(np.random.choice(shapes, remainder, replace=False))
            np.random.shuffle(shape_sequence)
        else:
            shape_sequence = [np.random.choice(shapes) for _ in range(n_samples)]
        
        for i in range(n_samples):
            shape = shape_sequence[i]
            lattice = np.random.uniform(*lattice_range)
            void = np.random.uniform(*void_range)
            
            if (i + 1) % 100 == 0 or (i + 1) % 10 == 0 and i < 50:
                print(f"Sample {i+1}/{n_samples}: {shape}, a={lattice:.1f}mm, φ={void:.2f}")
            
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
                'physics_model': 'resonant_cavity'  # Mark the model type
            }
            
            dataset.append(sample)
            
            if (i + 1) % 100 == 0 or (i + 1) % 10 == 0 and i < 50:
                if bg_center:
                    print(f"   ✓ Resonance: {bg_center:.0f} Hz")
                else:
                    print(f"   ✗ No resonance")
        
        print(f"\n{'='*70}")
        print("DATASET COMPLETE")
        print(f"{'='*70}")
        
        # Statistics
        with_bg = [s for s in dataset if s['bandgap_center']]
        if with_bg:
            centers = [s['bandgap_center'] for s in with_bg]
            lattices = [s['lattice_constant'] for s in with_bg]
            
            print(f"\n📊 Dataset Statistics:")
            print(f"   Total: {len(dataset)}, With resonances: {len(with_bg)} ({100*len(with_bg)/len(dataset):.1f}%)")
            print(f"   Frequency range: {min(centers):.0f} - {max(centers):.0f} Hz")
            print(f"   Spread: {max(centers)-min(centers):.0f} Hz")
            
            correlation = np.corrcoef(lattices, centers)[0, 1]
            print(f"   Lattice-Freq correlation: {correlation:.3f}")
            
            print(f"\n   Shape distribution:")
            for shape in shapes:
                shape_samples = [s for s in with_bg if s['shape'] == shape]
                if shape_samples:
                    shape_freqs = [s['bandgap_center'] for s in shape_samples]
                    print(f"      {shape:10s}: {len(shape_samples):3d} samples, mean: {np.mean(shape_freqs):6.0f} Hz")
        
        print(f"{'='*70}")
        return dataset
    
    def visualize_sample(self, sample, save_path=None):
        """Visualization"""
        fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1.5, 1])
        
        ax1 = fig.add_subplot(gs[0])
        geometry = np.array(sample['geometry'])
        im = ax1.imshow(geometry, cmap='RdYlBu_r', interpolation='nearest')
        ax1.set_title(f"{sample['shape'].capitalize()}\na={sample['lattice_constant']:.1f}mm, φ={sample['void_ratio']:.2f}",
                     fontsize=11, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im, ax=ax1, fraction=0.046)
        
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
            ax2.annotate(f"{center:.0f} Hz",
                        xy=(center, 0.25), xytext=(center, 0.6),
                        bbox=dict(boxstyle='round', facecolor='wheat'),
                        ha='center', fontsize=9,
                        arrowprops=dict(arrowstyle='->', color='red'))
        
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Transmission')
        ax2.set_title('Resonant Cavity Transmission', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        ax3 = fig.add_subplot(gs[2])
        ax3.axis('off')
        
        text = f"Cavity Resonator\n{'='*20}\n\n"
        text += f"Lattice: {sample['lattice_constant']:.1f} mm\n"
        text += f"Void: {sample['void_ratio']:.2f}\n"
        text += f"Shape: {sample['shape']}\n\n"
        
        if sample['bandgap_center']:
            text += f"Resonance: {sample['bandgap_center']:.0f} Hz\n"
            text += f"Bandwidth: {sample['bandgap_width']:.0f} Hz\n"
            text += f"Depth: {sample['bandgap_depth']:.2f}\n"
        else:
            text += "No resonance\n"
        
        text += f"\nModel: Helmholtz\nresonator"
        
        ax3.text(0.1, 0.9, text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"      💾 Saved: {save_path}")
        
        plt.close()


if __name__ == "__main__":
    import os
    
    simulator = ResonantCavitySimulator(resolution=64)
    
    print("\n🎯 RESONANT CAVITY MODEL (V2.0)")
    print("="*70)
    print("Key difference from V1.0:")
    print("  V1.0: Bragg scattering (periodic interference)")
    print("  V2.0: Helmholtz resonator (cavity resonance)")
    print("="*70)
    
    dataset = simulator.generate_dataset(
        n_samples=2000,
        lattice_range=(5, 50),
        void_range=(0.2, 0.8),
        ensure_equal_shapes=True
    )
    
    output_path = "/root/shared/resonant_cavity_dataset.json" if os.path.exists("/root/shared") else "resonant_cavity_dataset.json"
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✅ Saved: {output_path}")
    
    print("\n🎨 Creating visualizations...")
    shapes_seen = set()
    for sample in dataset:
        shape = sample['shape']
        if shape not in shapes_seen:
            shapes_seen.add(shape)
            filename_base = f"resonant_cavity_{shape}.png"
            filename = f"/root/shared/{filename_base}" if os.path.exists("/root/shared") else filename_base
            simulator.visualize_sample(sample, save_path=filename)
    
    print(f"\n🎯 Resonant cavity dataset ready!")
    print(f"   This uses DIFFERENT physics from V1.0 Bragg model")