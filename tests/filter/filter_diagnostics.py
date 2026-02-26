"""
Kalman Filter Diagnostic Tests

This script provides comprehensive tests to evaluate if the EKF is working properly
and identify potential failure points.

Tests included:
1. Innovation (Residual) Analysis - Check if innovations are zero-mean and consistent
2. Normalized Innovation Squared (NIS) Test - Statistical consistency check
3. Covariance Matrix Health - Positive definiteness, symmetry, condition number
4. Filter Divergence Detection - Track if covariance grows unbounded
5. Measurement Rejection Rate - How often measurements are rejected
6. State Observability - Check if all states are being updated
7. Consistency Checks - Compare predicted vs. actual uncertainty
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import sqrtm


class KalmanFilterDiagnostics:
    """Diagnostic test suite for Kalman filter validation."""
    
    def __init__(self, data_dir):
        """Load data from the specified directory.
        
        Args:
            data_dir: Path to directory containing state_estimator.csv, 
                     uwb_positions.csv, and ekf_covariance.txt
        """
        self.data_dir = Path(data_dir)
        self.load_data()
        
    def load_data(self):
        """Load all required data files."""
        print("Loading data files...")
        
        # Load state estimator data
        state_path = self.data_dir / 'state_estimator.csv'
        if not state_path.exists():
            state_path = self.data_dir / 'kalman.csv'
        self.state_df = pd.read_csv(state_path)
        
        # Load UWB measurements
        uwb_path = self.data_dir / 'uwb_positions.csv'
        if uwb_path.exists():
            self.uwb_df = pd.read_csv(uwb_path)
            # Detect format: new format has 'tag_id', old has 'x1', 'y1', 'x2', 'y2'
            if 'tag_id' in self.uwb_df.columns:
                # Convert new format to old format for compatibility
                self._convert_uwb_format_new_to_old()
        else:
            self.uwb_df = None
            print("Warning: No UWB data found")
        
        # Load covariance data if available
        cov_path = self.data_dir / 'ekf_covariance.txt'
        if cov_path.exists():
            self.load_covariance_data(cov_path)
        else:
            self.covariances = None
            self.cov_timestamps = None
            print("Warning: No covariance data found")
        
        print(f"Loaded {len(self.state_df)} state samples")
        if self.uwb_df is not None:
            print(f"Loaded {len(self.uwb_df)} UWB measurements")
    
    def _convert_uwb_format_new_to_old(self):
        """Convert new UWB format (tag_id) to old format (x1, y1, x2, y2)."""
        # Group by timestamp
        grouped = self.uwb_df.groupby('timestamp')
        
        rows = []
        for timestamp, group in grouped:
            row = {'timestamp': timestamp}
            
            tag0 = group[group['tag_id'] == 0]
            tag1 = group[group['tag_id'] == 1]
            
            if not tag0.empty:
                row['x1'] = tag0.iloc[0]['x']
                row['y1'] = tag0.iloc[0]['y']
            else:
                row['x1'] = np.nan
                row['y1'] = np.nan
            
            if not tag1.empty:
                row['x2'] = tag1.iloc[0]['x']
                row['y2'] = tag1.iloc[0]['y']
            else:
                row['x2'] = np.nan
                row['y2'] = np.nan
            
            rows.append(row)
        
        self.uwb_df = pd.DataFrame(rows)
    
    def load_covariance_data(self, cov_path):
        """Parse the covariance text file into structured data."""
        covariances = []
        timestamps = []
        
        with open(cov_path, 'r') as f:
            current_cov = []
            current_timestamp = None
            
            for line in f:
                line = line.strip()
                if line.startswith('# timestamp:'):
                    if current_cov and current_timestamp is not None:
                        covariances.append(np.array(current_cov))
                        timestamps.append(current_timestamp)
                    current_cov = []
                    current_timestamp = float(line.split(':')[1].strip())
                elif line and not line.startswith('#'):
                    # Parse covariance matrix row
                    row = [float(x) for x in line.split()]
                    current_cov.append(row)
            
            # Add last matrix
            if current_cov and current_timestamp is not None:
                covariances.append(np.array(current_cov))
                timestamps.append(current_timestamp)
        
        self.covariances = covariances
        self.cov_timestamps = np.array(timestamps)
        print(f"Loaded {len(covariances)} covariance matrices")
    
    def test_innovation_analysis(self):
        """Test 1: Analyze innovation (residual) statistics.
        
        Innovation = measurement - prediction
        Should be zero-mean with covariance matching the innovation covariance S.
        """
        print("\n" + "="*70)
        print("TEST 1: INNOVATION ANALYSIS")
        print("="*70)
        
        if self.uwb_df is None:
            print("SKIP: No UWB measurement data available")
            return None
        
        # Calculate innovations: UWB measurement - state estimate
        # Need to interpolate state to UWB measurement times
        innovations = []
        
        for _, uwb_row in self.uwb_df.iterrows():
            timestamp = uwb_row['timestamp']
            
            # Find closest state estimate
            time_diffs = np.abs(self.state_df['timestamp'] - timestamp)
            closest_idx = time_diffs.argmin()
            
            if time_diffs.iloc[closest_idx] < 0.05:  # Within 50ms
                state_row = self.state_df.iloc[closest_idx]
                
                # Calculate innovation for this measurement
                # Average both tags if available
                x_meas = None
                y_meas = None
                
                if pd.notna(uwb_row['x1']) and pd.notna(uwb_row['y1']):
                    if pd.notna(uwb_row['x2']) and pd.notna(uwb_row['y2']):
                        # Both tags available - average them
                        x_meas = (uwb_row['x1'] + uwb_row['x2']) / 2
                        y_meas = (uwb_row['y1'] + uwb_row['y2']) / 2
                    else:
                        # Only tag 1 available
                        x_meas = uwb_row['x1']
                        y_meas = uwb_row['y1']
                elif pd.notna(uwb_row['x2']) and pd.notna(uwb_row['y2']):
                    # Only tag 2 available
                    x_meas = uwb_row['x2']
                    y_meas = uwb_row['y2']
                
                if x_meas is not None and y_meas is not None:
                    innov_x = x_meas - state_row['px']
                    innov_y = y_meas - state_row['py']
                    innovations.append([innov_x, innov_y])
        
        if not innovations:
            print("FAIL: No matching UWB-state pairs found")
            return None
        
        innovations = np.array(innovations)
        
        # Statistics
        mean_innov = np.mean(innovations, axis=0)
        std_innov = np.std(innovations, axis=0)
        
        print(f"Number of innovations: {len(innovations)}")
        print(f"Innovation mean (should be ~0): x={mean_innov[0]:.4f}m, y={mean_innov[1]:.4f}m")
        print(f"Innovation std dev: x={std_innov[0]:.4f}m, y={std_innov[1]:.4f}m")
        
        # Zero-mean test (t-test)
        t_stat_x, p_val_x = stats.ttest_1samp(innovations[:, 0], 0)
        t_stat_y, p_val_y = stats.ttest_1samp(innovations[:, 1], 0)
        
        alpha = 0.05
        print(f"\nZero-mean test (H0: mean=0, alpha={alpha}):")
        print(f"  X: p-value={p_val_x:.4f} {'PASS' if p_val_x > alpha else 'FAIL (biased)'}")
        print(f"  Y: p-value={p_val_y:.4f} {'PASS' if p_val_y > alpha else 'FAIL (biased)'}")
        
        # Normality test
        _, p_norm_x = stats.normaltest(innovations[:, 0])
        _, p_norm_y = stats.normaltest(innovations[:, 1])
        print(f"\nNormality test (should be normal distribution):")
        print(f"  X: p-value={p_norm_x:.4f} {'PASS' if p_norm_x > alpha else 'FAIL'}")
        print(f"  Y: p-value={p_norm_y:.4f} {'PASS' if p_norm_y > alpha else 'FAIL'}")
        
        return innovations
    
    def test_nis_consistency(self):
        """Test 2: Normalized Innovation Squared (NIS) Test.
        
        NIS = innovation^T * S^-1 * innovation
        Should follow chi-squared distribution with df = measurement dimension.
        
        This tests if the filter's uncertainty (covariance) is consistent with
        the actual errors.
        """
        print("\n" + "="*70)
        print("TEST 2: NORMALIZED INNOVATION SQUARED (NIS) TEST")
        print("="*70)
        
        if self.uwb_df is None or self.covariances is None:
            print("SKIP: Requires both UWB and covariance data")
            return None
        
        nis_values = []
        
        # UWB measurement has dimension 3 (x, y, z) but we'll use 2D (x, y)
        measurement_dim = 2
        
        # R matrix for UWB (from filter: R_uwb_range = 0.1^2)
        R_uwb = np.eye(2) * (0.1 ** 2)
        
        for _, uwb_row in self.uwb_df.iterrows():
            timestamp = uwb_row['timestamp']
            
            # Find closest state and covariance
            time_diffs = np.abs(self.state_df['timestamp'] - timestamp)
            closest_idx = time_diffs.argmin()
            
            if time_diffs.iloc[closest_idx] > 0.05:
                continue
            
            state_row = self.state_df.iloc[closest_idx]
            
            # Calculate averaged measurement
            x_meas = None
            y_meas = None
            
            if pd.notna(uwb_row['x1']) and pd.notna(uwb_row['y1']):
                if pd.notna(uwb_row['x2']) and pd.notna(uwb_row['y2']):
                    x_meas = (uwb_row['x1'] + uwb_row['x2']) / 2
                    y_meas = (uwb_row['y1'] + uwb_row['y2']) / 2
                else:
                    x_meas = uwb_row['x1']
                    y_meas = uwb_row['y1']
            elif pd.notna(uwb_row['x2']) and pd.notna(uwb_row['y2']):
                x_meas = uwb_row['x2']
                y_meas = uwb_row['y2']
            
            if x_meas is None or y_meas is None:
                continue
            
            # Find covariance matrix
            cov_time_diffs = np.abs(self.cov_timestamps - timestamp)
            cov_idx = cov_time_diffs.argmin()
            
            if cov_time_diffs[cov_idx] > 0.05:
                continue
            
            P = self.covariances[cov_idx]
            
            # Innovation
            y = np.array([
                x_meas - state_row['px'],
                y_meas - state_row['py']
            ])
            
            # H matrix for position measurement (from error-state)
            # Measures position directly: H = [I_2x2, 0_2x7]
            H = np.zeros((2, 9))
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            
            # Innovation covariance: S = H*P*H^T + R
            S = H @ P @ H.T + R_uwb
            
            try:
                S_inv = np.linalg.inv(S)
                nis = y.T @ S_inv @ y
                nis_values.append(nis)
            except np.linalg.LinAlgError:
                pass
        
        if not nis_values:
            print("FAIL: No NIS values computed")
            return None
        
        nis_values = np.array(nis_values)
        
        # Chi-squared test
        # For dimension 2, 95% of NIS values should be below chi2(0.95, 2) = 5.99
        chi2_95 = stats.chi2.ppf(0.95, measurement_dim)
        chi2_05 = stats.chi2.ppf(0.05, measurement_dim)
        
        percent_in_95 = np.sum(nis_values <= chi2_95) / len(nis_values) * 100
        percent_below_05 = np.sum(nis_values <= chi2_05) / len(nis_values) * 100
        
        print(f"Number of NIS samples: {len(nis_values)}")
        print(f"Mean NIS: {np.mean(nis_values):.4f} (expected: {measurement_dim})")
        print(f"NIS range: [{np.min(nis_values):.4f}, {np.max(nis_values):.4f}]")
        print(f"\nChi-squared bounds (df={measurement_dim}):")
        print(f"  5% bound: {chi2_05:.2f}")
        print(f"  95% bound: {chi2_95:.2f}")
        print(f"\nConsistency check:")
        print(f"  {percent_in_95:.1f}% below 95% bound (expect ~95%): {'PASS' if 90 < percent_in_95 < 98 else 'FAIL'}")
        print(f"  {percent_below_05:.1f}% below 5% bound (expect ~5%): {'PASS' if 2 < percent_below_05 < 10 else 'FAIL'}")
        
        if np.mean(nis_values) > measurement_dim * 2:
            print("\nWARNING: Mean NIS much higher than expected - filter may be overconfident")
        elif np.mean(nis_values) < measurement_dim * 0.5:
            print("\nWARNING: Mean NIS much lower than expected - filter may be too conservative")
        
        return nis_values
    
    def test_covariance_health(self):
        """Test 3: Check covariance matrix properties.
        
        Checks:
        - Positive definiteness (all eigenvalues > 0)
        - Symmetry
        - Condition number (numerical stability)
        - Reasonable bounds on diagonal elements
        """
        print("\n" + "="*70)
        print("TEST 3: COVARIANCE MATRIX HEALTH")
        print("="*70)
        
        if self.covariances is None:
            print("SKIP: No covariance data available")
            return None
        
        issues = []
        condition_numbers = []
        min_eigenvalues = []
        max_diagonals = []
        
        for i, P in enumerate(self.covariances):
            # Check symmetry
            if not np.allclose(P, P.T, rtol=1e-6):
                issues.append(f"Sample {i}: Matrix not symmetric")
            
            # Check positive definiteness via eigenvalues
            eigenvalues = np.linalg.eigvalsh(P)
            min_eig = np.min(eigenvalues)
            min_eigenvalues.append(min_eig)
            
            if min_eig <= 0:
                issues.append(f"Sample {i}: Not positive definite (min eig={min_eig:.2e})")
            
            # Condition number
            cond = np.linalg.cond(P)
            condition_numbers.append(cond)
            
            if cond > 1e10:
                issues.append(f"Sample {i}: Poor conditioning (cond={cond:.2e})")
            
            # Check diagonal bounds
            diag = np.diag(P)
            max_diag = np.max(diag)
            max_diagonals.append(max_diag)
            
            if max_diag > 100:
                issues.append(f"Sample {i}: Very large uncertainty (max diag={max_diag:.2f})")
        
        print(f"Analyzed {len(self.covariances)} covariance matrices")
        print(f"\nCondition number:")
        print(f"  Mean: {np.mean(condition_numbers):.2e}")
        print(f"  Max: {np.max(condition_numbers):.2e} {'FAIL (unstable)' if np.max(condition_numbers) > 1e10 else 'PASS'}")
        
        print(f"\nMinimum eigenvalue:")
        print(f"  Mean: {np.mean(min_eigenvalues):.2e}")
        print(f"  Min: {np.min(min_eigenvalues):.2e} {'FAIL (not positive definite)' if np.min(min_eigenvalues) <= 0 else 'PASS'}")
        
        print(f"\nMaximum diagonal element:")
        print(f"  Mean: {np.mean(max_diagonals):.4f}")
        print(f"  Max: {np.max(max_diagonals):.4f}")
        
        if issues:
            print(f"\n⚠ Found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
        else:
            print("\n✓ All covariance matrices are healthy")
        
        return {
            'condition_numbers': condition_numbers,
            'min_eigenvalues': min_eigenvalues,
            'max_diagonals': max_diagonals,
            'issues': issues
        }
    
    def test_filter_divergence(self):
        """Test 4: Detect filter divergence.
        
        Checks if the covariance trace is growing unbounded, which indicates
        the filter is diverging (losing track).
        """
        print("\n" + "="*70)
        print("TEST 4: FILTER DIVERGENCE DETECTION")
        print("="*70)
        
        if self.covariances is None:
            print("SKIP: No covariance data available")
            return None
        
        traces = [np.trace(P) for P in self.covariances]
        traces = np.array(traces)
        
        print(f"Covariance trace statistics:")
        print(f"  Initial: {traces[0]:.4f}")
        print(f"  Final: {traces[-1]:.4f}")
        print(f"  Mean: {np.mean(traces):.4f}")
        print(f"  Max: {np.max(traces):.4f}")
        
        # Check for monotonic growth (divergence indicator)
        # Use moving average to smooth
        window = min(100, len(traces) // 10)
        if window > 5:
            from scipy.ndimage import uniform_filter1d
            smoothed = uniform_filter1d(traces, size=window)
            
            # Check if generally increasing
            start_avg = np.mean(smoothed[:window])
            end_avg = np.mean(smoothed[-window:])
            growth_ratio = end_avg / start_avg if start_avg > 0 else float('inf')
            
            print(f"\nDivergence check (trace growth):")
            print(f"  Start average: {start_avg:.4f}")
            print(f"  End average: {end_avg:.4f}")
            print(f"  Growth ratio: {growth_ratio:.2f}x")
            
            if growth_ratio > 2.0:
                print("  WARNING: Covariance growing - possible divergence")
            elif growth_ratio < 0.5:
                print("  Note: Covariance decreasing - filter is converging")
            else:
                print("  PASS: Covariance stable")
        
        return traces
    
    def test_state_observability(self):
        """Test 5: Check if all state components are being updated.
        
        If a state component's variance doesn't decrease over time,
        it may not be observable from the measurements.
        """
        print("\n" + "="*70)
        print("TEST 5: STATE OBSERVABILITY")
        print("="*70)
        
        if self.covariances is None:
            print("SKIP: No covariance data available")
            return None
        
        # Track diagonal elements over time
        state_labels = ['px', 'py', 'pz', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw']
        
        n_samples = len(self.covariances)
        n_states = 9
        
        # Get initial and final variances
        P_init = self.covariances[min(10, n_samples-1)]  # After a few samples
        P_final = self.covariances[-1]
        
        diag_init = np.diag(P_init)
        diag_final = np.diag(P_final)
        
        print("Variance change (initial → final):")
        print(f"{'State':<10} {'Initial':>12} {'Final':>12} {'Change':>12} {'Status'}")
        print("-" * 60)
        
        observable_states = []
        unobservable_states = []
        
        for i, label in enumerate(state_labels):
            change = (diag_final[i] - diag_init[i]) / diag_init[i] * 100 if diag_init[i] > 0 else 0
            
            if abs(change) < 5:  # Less than 5% change
                status = "⚠ No update"
                unobservable_states.append(label)
            elif diag_final[i] < diag_init[i]:
                status = "✓ Converging"
                observable_states.append(label)
            else:
                status = "⚠ Growing"
            
            print(f"{label:<10} {diag_init[i]:>12.4f} {diag_final[i]:>12.4f} {change:>11.1f}% {status}")
        
        print(f"\nSummary:")
        print(f"  Observable states: {len(observable_states)}/{n_states}")
        print(f"  Unobservable/unchanged: {len(unobservable_states)}")
        
        if unobservable_states:
            print(f"  WARNING: States not being updated: {', '.join(unobservable_states)}")
        
        return {
            'observable': observable_states,
            'unobservable': unobservable_states
        }
    
    def test_measurement_update_rate(self):
        """Test 6: Analyze measurement update frequency.
        
        Checks how often measurements are arriving and if there are
        significant gaps that could affect filter performance.
        """
        print("\n" + "="*70)
        print("TEST 6: MEASUREMENT UPDATE RATE")
        print("="*70)
        
        if self.uwb_df is None:
            print("SKIP: No UWB data available")
            return None
        
        # Calculate inter-measurement times
        timestamps = self.uwb_df['timestamp'].values
        dt_meas = np.diff(timestamps)
        
        print(f"Measurement statistics:")
        print(f"  Total measurements: {len(timestamps)}")
        print(f"  Time span: {timestamps[-1] - timestamps[0]:.2f} seconds")
        print(f"  Mean dt: {np.mean(dt_meas):.4f} s ({1/np.mean(dt_meas):.1f} Hz)")
        print(f"  Median dt: {np.median(dt_meas):.4f} s")
        print(f"  Std dt: {np.std(dt_meas):.4f} s")
        print(f"  Max gap: {np.max(dt_meas):.4f} s")
        
        # Check for large gaps
        large_gaps = np.where(dt_meas > 0.1)[0]  # > 100ms gaps
        if len(large_gaps) > 0:
            print(f"\n  WARNING: {len(large_gaps)} measurement gaps > 100ms")
            print(f"  Largest gap: {np.max(dt_meas):.4f} s")
        else:
            print(f"\n  ✓ No significant measurement gaps detected")
        
        return dt_meas
    
    def run_all_tests(self):
        """Run all diagnostic tests and generate summary report."""
        print("\n" + "="*70)
        print("KALMAN FILTER DIAGNOSTIC TEST SUITE")
        print("="*70)
        print(f"Data directory: {self.data_dir}")
        
        results = {}
        
        # Run all tests
        results['innovations'] = self.test_innovation_analysis()
        results['nis'] = self.test_nis_consistency()
        results['covariance_health'] = self.test_covariance_health()
        results['divergence'] = self.test_filter_divergence()
        results['observability'] = self.test_state_observability()
        results['update_rate'] = self.test_measurement_update_rate()
        
        # Generate plots
        self.plot_diagnostics(results)
        
        print("\n" + "="*70)
        print("DIAGNOSTIC TESTS COMPLETE")
        print("="*70)
        
        return results
    
    def plot_diagnostics(self, results):
        """Generate diagnostic plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Kalman Filter Diagnostics', fontsize=16, fontweight='bold')
        
        # Plot 1: Innovation histogram
        if results['innovations'] is not None:
            ax = axes[0, 0]
            innovations = results['innovations']
            ax.hist(innovations[:, 0], bins=50, alpha=0.7, label='X', density=True)
            ax.hist(innovations[:, 1], bins=50, alpha=0.7, label='Y', density=True)
            ax.axvline(0, color='k', linestyle='--', linewidth=1)
            ax.set_xlabel('Innovation (m)')
            ax.set_ylabel('Density')
            ax.set_title('Innovation Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 2: NIS over time
        if results['nis'] is not None:
            ax = axes[0, 1]
            nis = results['nis']
            ax.plot(nis, alpha=0.6, linewidth=0.5)
            ax.axhline(stats.chi2.ppf(0.95, 2), color='r', linestyle='--', label='95% bound')
            ax.axhline(stats.chi2.ppf(0.05, 2), color='r', linestyle='--', label='5% bound')
            ax.axhline(2, color='g', linestyle='--', label='Expected mean')
            ax.set_xlabel('Measurement #')
            ax.set_ylabel('NIS')
            ax.set_title('Normalized Innovation Squared')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Condition number
        if results['covariance_health'] is not None:
            ax = axes[0, 2]
            cond_nums = results['covariance_health']['condition_numbers']
            ax.plot(cond_nums)
            ax.set_xlabel('Sample #')
            ax.set_ylabel('Condition Number')
            ax.set_title('Covariance Conditioning')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Covariance trace (divergence)
        if results['divergence'] is not None:
            ax = axes[1, 0]
            traces = results['divergence']
            ax.plot(traces)
            ax.set_xlabel('Sample #')
            ax.set_ylabel('Trace(P)')
            ax.set_title('Covariance Trace (Total Uncertainty)')
            ax.grid(True, alpha=0.3)
        
        # Plot 5: State variances over time
        if self.covariances is not None:
            ax = axes[1, 1]
            state_labels = ['px', 'py', 'pz', 'vx', 'vy', 'vz', 'r', 'p', 'y']
            for i in range(min(9, len(state_labels))):
                variances = [P[i, i] for P in self.covariances]
                ax.plot(variances, label=state_labels[i], alpha=0.7)
            ax.set_xlabel('Sample #')
            ax.set_ylabel('Variance')
            ax.set_title('State Variances Over Time')
            ax.set_yscale('log')
            ax.legend(ncol=3, fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Plot 6: Measurement update intervals
        if results['update_rate'] is not None:
            ax = axes[1, 2]
            dt_meas = results['update_rate']
            ax.hist(dt_meas * 1000, bins=50)  # Convert to ms
            ax.set_xlabel('Inter-measurement time (ms)')
            ax.set_ylabel('Count')
            ax.set_title('Measurement Update Intervals')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'filter_diagnostics.png', dpi=150)
        print(f"\nDiagnostic plots saved to: {self.data_dir / 'filter_diagnostics.png'}")


def main():
    """Run diagnostics on the most recent data."""
    # Find most recent data directory
    base_dir = Path(__file__).parent.parent / 'example_output'
    
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} not found")
        return
    
    # Get most recent subdirectory
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"Error: No data directories found in {base_dir}")
        return
    
    latest_dir = max(subdirs, key=lambda d: d.name)
    print(f"Using data from: {latest_dir}")
    
    # Run diagnostics
    diagnostics = KalmanFilterDiagnostics(latest_dir)
    results = diagnostics.run_all_tests()


if __name__ == '__main__':
    main()
