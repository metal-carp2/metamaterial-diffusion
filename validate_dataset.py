"""
Dataset Validation and Quality Check Script
Analyzes the generated dataset to ensure physics makes sense
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_dataset(filepath):
    """Load dataset from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)

def validate_physics(dataset):
    """
    Check if the physics relationships make sense
    """
    print("\n" + "="*70)
    print("PHYSICS VALIDATION")
    print("="*70)
    
    issues = []
    
    # Extract data
    samples_with_bg = [s for s in dataset if s['bandgap_center'] is not None]
    
    if len(samples_with_bg) == 0:
        print("❌ CRITICAL: No samples have bandgaps!")
        return False
    
    print(f"\n✅ {len(samples_with_bg)}/{len(dataset)} samples have bandgaps")
    
    # Check 1: Frequency vs Lattice Constant relationship
    # Physics: f ≈ c/(2*a), so larger lattice = lower frequency
    lattices = [s['lattice_constant'] for s in samples_with_bg]
    frequencies = [s['bandgap_center'] for s in samples_with_bg]
    
    correlation = np.corrcoef(lattices, frequencies)[0, 1]
    print(f"\n📊 Lattice vs Frequency Correlation: {correlation:.3f}")
    
    if correlation > -0.5:
        issues.append("⚠️  WARNING: Lattice and frequency should be negatively correlated")
        print(f"   Expected: < -0.5 (inverse relationship)")
        print(f"   Got: {correlation:.3f}")
        print(f"   Larger structures should block lower frequencies!")
    else:
        print(f"   ✅ Good! Negative correlation as expected")
    
    # Check 2: Frequency range diversity
    freq_min = min(frequencies)
    freq_max = max(frequencies)
    freq_range = freq_max - freq_min
    
    print(f"\n📊 Frequency Range: {freq_min:.0f} - {freq_max:.0f} Hz")
    print(f"   Range span: {freq_range:.0f} Hz ({freq_range/freq_min*100:.1f}% of minimum)")
    
    if freq_range < 500:
        issues.append("⚠️  WARNING: Frequency range is too narrow (<500 Hz)")
        print(f"   ⚠️  Too narrow! All designs clustering around same frequency")
    else:
        print(f"   ✅ Good diversity!")
    
    # Check 3: Shape differences
    print(f"\n📊 Shape Analysis:")
    shapes = set(s['shape'] for s in samples_with_bg)
    shape_freqs = {}
    
    for shape in shapes:
        shape_samples = [s for s in samples_with_bg if s['shape'] == shape]
        if len(shape_samples) > 0:
            avg_freq = np.mean([s['bandgap_center'] for s in shape_samples])
            shape_freqs[shape] = avg_freq
            print(f"   {shape:10s}: {avg_freq:6.0f} Hz (n={len(shape_samples)})")
    
    # Check if shapes have different frequencies
    if len(shape_freqs) > 1:
        freq_values = list(shape_freqs.values())
        freq_std = np.std(freq_values)
        freq_mean = np.mean(freq_values)
        variation = freq_std / freq_mean * 100
        
        print(f"\n   Shape variation: {variation:.1f}%")
        if variation < 5:
            issues.append("⚠️  WARNING: Shapes have very similar frequencies")
            print(f"   ⚠️  Shapes too similar! Expected >5% variation")
        else:
            print(f"   ✅ Good! Shapes produce different frequencies")
    
    # Check 4: Void ratio effects
    void_ratios = [s['void_ratio'] for s in samples_with_bg]
    void_correlation = np.corrcoef(void_ratios, frequencies)[0, 1]
    
    print(f"\n📊 Void Ratio vs Frequency Correlation: {void_correlation:.3f}")
    if abs(void_correlation) > 0.3:
        print(f"   ✅ Void ratio affects frequency (good!)")
    else:
        print(f"   ⚠️  Weak effect - void ratio may not be influencing results enough")
    
    # Check 5: Bandgap quality
    depths = [s['bandgap_depth'] for s in samples_with_bg]
    widths = [s['bandgap_width'] for s in samples_with_bg]
    
    print(f"\n📊 Bandgap Quality:")
    print(f"   Depth: {np.mean(depths):.3f} ± {np.std(depths):.3f}")
    print(f"   Width: {np.mean(widths):.0f} ± {np.std(widths):.0f} Hz")
    
    if np.mean(depths) < 0.3:
        issues.append("⚠️  WARNING: Bandgaps are too shallow")
        print(f"   ⚠️  Weak blocking! Mean depth < 0.3")
    else:
        print(f"   ✅ Good blocking capability")
    
    # Summary
    print(f"\n" + "="*70)
    if len(issues) == 0:
        print("✅ VALIDATION PASSED: Physics looks good!")
        return True
    else:
        print(f"⚠️  FOUND {len(issues)} ISSUES:")
        for issue in issues:
            print(f"   {issue}")
        print("\nSuggestions:")
        print("   1. Increase lattice constant range for more frequency diversity")
        print("   2. Check shape factor calculations")
        print("   3. Adjust bandwidth calculation in simulator")
        return False

def create_diagnostic_plots(dataset, output_dir="/root/shared/"):
    """
    Create diagnostic plots to visualize relationships
    """
    print(f"\n" + "="*70)
    print("CREATING DIAGNOSTIC PLOTS")
    print("="*70)
    
    samples_with_bg = [s for s in dataset if s['bandgap_center'] is not None]
    
    if len(samples_with_bg) == 0:
        print("❌ No samples with bandgaps to plot")
        return
    
    # Extract data
    lattices = np.array([s['lattice_constant'] for s in samples_with_bg])
    frequencies = np.array([s['bandgap_center'] for s in samples_with_bg])
    void_ratios = np.array([s['void_ratio'] for s in samples_with_bg])
    shapes = [s['shape'] for s in samples_with_bg]
    depths = np.array([s['bandgap_depth'] for s in samples_with_bg])
    widths = np.array([s['bandgap_width'] for s in samples_with_bg])
    
    # Create comprehensive diagnostic figure
    fig = plt.figure(figsize=(18, 10))
    
    # Plot 1: Frequency vs Lattice Constant
    ax1 = plt.subplot(2, 3, 1)
    
    # Color by shape
    shape_colors = {'circle': 'blue', 'square': 'red', 'hexagon': 'green', 
                   'cross': 'orange', 'star': 'purple'}
    for shape in set(shapes):
        mask = np.array(shapes) == shape
        ax1.scatter(lattices[mask], frequencies[mask], 
                   c=shape_colors.get(shape, 'gray'), 
                   label=shape, alpha=0.7, s=100)
    
    # Expected relationship line (f = c/(2*a))
    a_range = np.linspace(lattices.min(), lattices.max(), 100)
    f_expected = 343 / (2 * a_range * 1e-3)  # Air speed of sound
    ax1.plot(a_range, f_expected, 'k--', alpha=0.5, label='Theory: c/(2a)')
    
    ax1.set_xlabel('Lattice Constant (mm)', fontsize=12)
    ax1.set_ylabel('Bandgap Center Frequency (Hz)', fontsize=12)
    ax1.set_title('Frequency vs Lattice Constant\n(Should follow inverse relationship)', 
                 fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Frequency distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(frequencies, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(frequencies), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(frequencies):.0f} Hz')
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Frequency Distribution\n(Should be spread out)', 
                 fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Frequency vs Void Ratio
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(void_ratios, frequencies, c=lattices, cmap='viridis', s=100, alpha=0.7)
    cbar = plt.colorbar(ax3.scatter(void_ratios, frequencies, c=lattices, 
                                    cmap='viridis', s=100, alpha=0.7))
    cbar.set_label('Lattice Constant (mm)', fontsize=11)
    ax3.set_xlabel('Void Ratio', fontsize=12)
    ax3.set_ylabel('Frequency (Hz)', fontsize=12)
    ax3.set_title('Frequency vs Void Ratio\n(Colored by lattice constant)', 
                 fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Shape comparison boxplot
    ax4 = plt.subplot(2, 3, 4)
    shape_data = []
    shape_labels = []
    for shape in sorted(set(shapes)):
        shape_freqs = frequencies[np.array(shapes) == shape]
        if len(shape_freqs) > 0:
            shape_data.append(shape_freqs)
            shape_labels.append(shape)
    
    bp = ax4.boxplot(shape_data, labels=shape_labels, patch_artist=True)
    for patch, shape in zip(bp['boxes'], shape_labels):
        patch.set_facecolor(shape_colors.get(shape, 'gray'))
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Frequency (Hz)', fontsize=12)
    ax4.set_title('Frequency by Shape\n(Different shapes should have different ranges)', 
                 fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    
    # Plot 5: Bandgap quality
    ax5 = plt.subplot(2, 3, 5)
    scatter = ax5.scatter(widths, depths, c=frequencies, cmap='plasma', s=100, alpha=0.7)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Frequency (Hz)', fontsize=11)
    ax5.set_xlabel('Bandgap Width (Hz)', fontsize=12)
    ax5.set_ylabel('Bandgap Depth', fontsize=12)
    ax5.set_title('Bandgap Quality\n(Larger = better)', 
                 fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = "DATASET SUMMARY\n" + "="*35 + "\n\n"
    stats_text += f"Total samples: {len(dataset)}\n"
    stats_text += f"With bandgaps: {len(samples_with_bg)}\n"
    stats_text += f"Success rate: {len(samples_with_bg)/len(dataset)*100:.1f}%\n\n"
    
    stats_text += f"Frequency Range:\n"
    stats_text += f"  Min: {frequencies.min():.0f} Hz\n"
    stats_text += f"  Max: {frequencies.max():.0f} Hz\n"
    stats_text += f"  Mean: {frequencies.mean():.0f} Hz\n"
    stats_text += f"  Std: {frequencies.std():.0f} Hz\n\n"
    
    stats_text += f"Lattice Constant:\n"
    stats_text += f"  Range: {lattices.min():.1f} - {lattices.max():.1f} mm\n"
    stats_text += f"  Mean: {lattices.mean():.1f} mm\n\n"
    
    stats_text += f"Bandgap Quality:\n"
    stats_text += f"  Depth: {depths.mean():.3f} ± {depths.std():.3f}\n"
    stats_text += f"  Width: {widths.mean():.0f} ± {widths.std():.0f} Hz\n\n"
    
    stats_text += f"Correlations:\n"
    stats_text += f"  Lattice-Freq: {np.corrcoef(lattices, frequencies)[0,1]:.3f}\n"
    stats_text += f"  Void-Freq: {np.corrcoef(void_ratios, frequencies)[0,1]:.3f}\n"
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    output_file = output_dir + "diagnostic_plots.png"  # Always same filename
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved diagnostic plots to: diagnostic_plots.png (overwrites old version)")
    plt.close()

def compare_to_theory(dataset):
    """
    Compare results to theoretical Bragg scattering predictions
    """
    print(f"\n" + "="*70)
    print("THEORY COMPARISON")
    print("="*70)
    
    samples_with_bg = [s for s in dataset if s['bandgap_center'] is not None]
    
    errors = []
    for s in samples_with_bg:
        # Theoretical Bragg frequency
        a = s['lattice_constant'] * 1e-3  # Convert to meters
        f_theory = 343 / (2 * a)  # Air speed of sound
        
        # Actual frequency
        f_actual = s['bandgap_center']
        
        # Error
        error = abs(f_actual - f_theory)
        rel_error = error / f_theory * 100
        errors.append(rel_error)
    
    print(f"\nTheoretical vs Actual Comparison:")
    print(f"  Mean relative error: {np.mean(errors):.1f}%")
    print(f"  Std deviation: {np.std(errors):.1f}%")
    print(f"  Min error: {np.min(errors):.1f}%")
    print(f"  Max error: {np.max(errors):.1f}%")
    
    if np.mean(errors) > 50:
        print(f"\n⚠️  WARNING: Large deviation from theory!")
        print(f"   Shape factors and corrections may be too aggressive")
    elif np.mean(errors) < 10:
        print(f"\n⚠️  WARNING: Too close to pure Bragg theory!")
        print(f"   Shape effects may not be strong enough")
    else:
        print(f"\n✅ Good! Deviations are reasonable (10-50%)")
        print(f"   Shape-dependent corrections are working")

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("DATASET VALIDATION TOOL")
    print("="*70)
    
    # Load dataset
    dataset_file = "/root/shared/fenics_dataset_test.json"
    print(f"\nLoading dataset from: {dataset_file}")
    
    try:
        dataset = load_dataset(dataset_file)
        print(f"✅ Loaded {len(dataset)} samples")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {dataset_file}")
        print("   Run the simulator first to generate data!")
        exit(1)
    
    # Run validation
    is_valid = validate_physics(dataset)
    
    # Create diagnostic plots
    create_diagnostic_plots(dataset)
    
    # Compare to theory
    compare_to_theory(dataset)
    
    # Final recommendation
    print(f"\n" + "="*70)
    print("FINAL RECOMMENDATION")
    print("="*70)
    
    if is_valid:
        print("✅ Dataset quality is GOOD - ready for model training!")
        print("\nNext steps:")
        print("   1. Generate full dataset (2000+ samples)")
        print("   2. Train diffusion model on this data")
        print("   3. Test generation with validation loop")
    else:
        print("⚠️  Dataset has issues - consider adjustments")
        print("\nRecommended fixes:")
        print("   1. Adjust shape factors in simulator")
        print("   2. Increase lattice constant range")
        print("   3. Check bandwidth calculation")
        print("   4. Re-generate dataset and validate again")
    
    print(f"\n📊 Check diagnostic_plots.png for visual analysis")
    print("="*70)
    
    # Save validation summary to JSON
    with_bg = [s for s in dataset if s['bandgap_center'] is not None]
    
    validation_summary = {
        "total_samples": len(dataset),
        "samples_with_bandgaps": len(with_bg),
        "success_rate": len(with_bg) / len(dataset) if len(dataset) > 0 else 0,
    }
    
    if with_bg:
        centers = [s['bandgap_center'] for s in with_bg]
        lattices = [s['lattice_constant'] for s in with_bg]
        frequencies = centers
        void_ratios = [s['void_ratio'] for s in with_bg]
        depths = [s['bandgap_depth'] for s in with_bg]
        widths = [s['bandgap_width'] for s in with_bg]
        
        validation_summary.update({
            "frequency_stats": {
                "min": float(np.min(centers)),
                "max": float(np.max(centers)),
                "mean": float(np.mean(centers)),
                "std": float(np.std(centers)),
                "range": float(np.max(centers) - np.min(centers))
            },
            "lattice_stats": {
                "min": float(np.min(lattices)),
                "max": float(np.max(lattices)),
                "mean": float(np.mean(lattices))
            },
            "bandgap_quality": {
                "mean_depth": float(np.mean(depths)),
                "std_depth": float(np.std(depths)),
                "mean_width": float(np.mean(widths)),
                "std_width": float(np.std(widths))
            },
            "correlations": {
                "lattice_frequency": float(np.corrcoef(lattices, frequencies)[0, 1]),
                "void_frequency": float(np.corrcoef(void_ratios, frequencies)[0, 1])
            },
            "shape_breakdown": {}
        })
        
        # Add per-shape statistics
        shapes = [s['shape'] for s in with_bg]
        for shape in set(shapes):
            shape_samples = [s for s in with_bg if s['shape'] == shape]
            shape_freqs = [s['bandgap_center'] for s in shape_samples]
            validation_summary["shape_breakdown"][shape] = {
                "count": len(shape_samples),
                "mean_frequency": float(np.mean(shape_freqs)),
                "std_frequency": float(np.std(shape_freqs)) if len(shape_freqs) > 1 else 0
            }
        
        # Theory comparison
        errors = []
        for s in with_bg:
            a = s['lattice_constant'] * 1e-3
            f_theory = 343 / (2 * a)
            f_actual = s['bandgap_center']
            error = abs(f_actual - f_theory)
            rel_error = error / f_theory * 100
            errors.append(rel_error)
        
        validation_summary["theory_comparison"] = {
            "mean_relative_error_percent": float(np.mean(errors)),
            "std_relative_error_percent": float(np.std(errors)),
            "min_error_percent": float(np.min(errors)),
            "max_error_percent": float(np.max(errors))
        }
        
        validation_summary["validation_status"] = "PASSED" if is_valid else "FAILED"
    else:
        validation_summary["validation_status"] = "FAILED - NO BANDGAPS"
    
    # Save JSON
    json_output = "/root/shared/validation_summary.json"
    with open(json_output, 'w') as f:
        json.dump(validation_summary, f, indent=2)
    
    print(f"\n💾 Validation summary saved to: validation_summary.json")
    print("="*70)