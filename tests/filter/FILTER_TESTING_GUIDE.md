# Kalman Filter Testing Guide

This guide explains how to test your Extended Kalman Filter and interpret the results.

## Running the Tests

```bash
cd tests
python filter_diagnostics.py
```

The script will automatically analyze the most recent data from `example_output/`.

---

## What Each Test Checks

### Test 1: Innovation Analysis

**What it tests:** Whether the measurement residuals (innovation = measurement - prediction) are well-behaved.

**What to look for:**

- **Mean near zero**: Innovations should average to zero. If not, the filter has a systematic bias.
- **Normal distribution**: Residuals should follow a bell curve.
- **Zero-mean p-value > 0.05**: Statistical test that mean is zero.

**Common failure modes:**

- **High mean**: Filter is consistently over/under-estimating
- **Non-normal distribution**: Model assumptions violated (e.g., wrong noise model)
- **Very large std dev**: Measurements very noisy or filter not trusting them enough

---

### Test 2: NIS (Normalized Innovation Squared) Test

**What it tests:** Whether the filter's confidence (covariance) matches the actual errors.

**What to look for:**

- **Mean NIS ≈ measurement dimension** (2 for 2D position)
- **~95% of NIS values below the 95% bound** (5.99 for 2D measurements)
- **~5% of NIS values below the 5% bound** (0.10 for 2D measurements)

**Common failure modes:**

- **Mean NIS >> expected**: Filter is **overconfident** (covariance too small). Increase process noise `Qc` or measurement noise `R`.
- **Mean NIS << expected**: Filter is **too conservative** (covariance too large). Decrease process noise.
- **All NIS values very high**: Measurement model is wrong (e.g., wrong H matrix, units mismatch)

---

### Test 3: Covariance Matrix Health

**What it tests:** Mathematical properties of the covariance matrix.

**What to look for:**

- **All eigenvalues > 0**: Matrix is positive definite (required for Kalman filtering)
- **Condition number < 1e10**: Matrix is numerically stable
- **Symmetric**: Matrix should equal its transpose

**Common failure modes:**

- **Negative eigenvalues**: Numerical issues, likely from:
  - Not using Joseph form update
  - Roundoff errors
  - Process/measurement noise matrices not positive definite
- **High condition number**: Near-singular matrix, could cause:
  - Numerical instability
  - Inversion failures
  - Need to tune noise parameters

---

### Test 4: Filter Divergence Detection

**What it tests:** Whether the filter is "losing track" over time.

**What to look for:**

- **Covariance trace staying bounded or decreasing**
- **Growth ratio < 2x** from start to end

**Common failure modes:**

- **Unbounded growth**: Filter diverging, usually because:
  - Process noise `Qc` too high
  - Not enough measurements
  - Model mismatch (dynamics don't match reality)
  - Measurements being rejected
- **Rapid growth then plateau**: Initial convergence issues, may be OK if stabilizes

---

### Test 5: State Observability

**What it tests:** Whether measurements actually constrain all state variables.

**What to look for:**

- **Variance decreasing for observable states**
- **All relevant states showing "Converging" status**

**Common failure modes:**

- **Velocity not updating**: No encoder or insufficient motion to estimate velocity
- **Z-position not updating**: UWB doesn't constrain height well
- **Attitude not updating**: No IMU measurements in log
- **Some states stuck**: These states aren't observable from your sensor suite

**Implications:**

- Unobservable states will drift
- Consider adding sensors or constraints
- May need to fix certain states if not observable

---

### Test 6: Measurement Update Rate

**What it tests:** How frequently measurements arrive.

**What to look for:**

- **Regular measurement intervals**
- **No large gaps** (> 100ms)
- **Update rate >> prediction rate** (ideally 10+ Hz)

**Common failure modes:**

- **Irregular updates**: Sensor dropping measurements
- **Large gaps**: Filter has to extrapolate, accumulates error
- **Very slow rate**: Not enough information to correct drift

---

## Interpreting the Plots

### Innovation Distribution

- Should look like normal (bell curve) centered at zero
- Heavy tails → outlier measurements
- Skewed → systematic bias

### NIS Over Time

- Should bounce around the expected mean (green line)
- ~95% below red upper line
- Consistent high values → overconfident filter
- Consistent low values → too conservative

### Condition Number

- Should stay reasonably constant
- Spikes indicate numerical problems
- Growing trend → matrix becoming ill-conditioned

### Covariance Trace

- Total uncertainty in the filter
- Should decrease initially (filter converging)
- Then stabilize at steady-state value
- Growing trace → divergence

### State Variances

- Each line is one state variable's uncertainty
- Should decrease when measurements arrive
- Then bounce around steady-state
- Not decreasing → state not observable

### Measurement Intervals

- Should be concentrated around one value
- Wide spread → irregular sensor timing
- Long tail → occasional data dropouts

---

## Common Issues and Solutions

### Issue: Filter Overconfident (High NIS)

**Symptoms:** NIS >> 2, innovations larger than predicted
**Solutions:**

- Increase measurement noise `R_uwb_range`
- Increase process noise `Qc`
- Check for model mismatch

### Issue: Filter Too Conservative (Low NIS)

**Symptoms:** NIS << 2, very small innovations
**Solutions:**

- Decrease process noise `Qc`
- Decrease measurement noise `R`
- Filter may be correct - check actual errors

### Issue: Filter Diverging

**Symptoms:** Covariance trace growing unbounded
**Solutions:**

- Decrease process noise
- Increase measurement rate
- Check prediction model (bicycle kinematics)
- Verify measurements arriving

### Issue: Biased Estimates

**Symptoms:** Non-zero innovation mean, position drift
**Solutions:**

- Check for sensor calibration errors (UWB anchor positions)
- Verify coordinate frame consistency
- Check tag offset parameters
- Look for systematic measurement bias

### Issue: Poor Conditioning

**Symptoms:** Covariance matrix ill-conditioned
**Solutions:**

- Ensure using Joseph form covariance update (✓ you are)
- Check that Q and R matrices are positive definite
- May need to rescale states (e.g., use different units)

### Issue: States Not Converging

**Symptoms:** Variances not decreasing
**Solutions:**

- Check if state is observable from measurements
- Add more sensor types if needed
- May need to constrain unobservable states
- Verify measurements updating that state

---

## Recommended Workflow

1. **Run diagnostics on new data**

   ```bash
   python tests/filter_diagnostics.py
   ```

2. **Check innovation analysis first**
   - If biased, fix measurement issues before tuning

3. **Check NIS test**
   - Tune Q and R to get NIS in acceptable range

4. **Monitor for divergence**
   - Ensure covariance trace bounded

5. **Verify observability**
   - Confirm all important states being updated

6. **Iterate**
   - Collect more data, re-run tests
   - Fine-tune noise parameters
   - Validate changes improved consistency

---

## Quick Health Check

✅ **Healthy Filter:**

- Innovation mean ≈ 0
- Mean NIS ≈ measurement dimension
- Covariance trace bounded
- All critical states converging
- No numerical issues

⚠️ **Needs Tuning:**

- NIS consistently high/low (adjust Q, R)
- Some states not converging (may be OK if not critical)

🔴 **Critical Issues:**

- Filter diverging (unbounded covariance)
- Numerical errors (non-positive-definite, high condition number)
- Large systematic bias (non-zero innovation mean)
- Covariance not decreasing at all

---

## Advanced: Manual NIS Calculation

If you want to manually verify NIS for specific measurements:

```python
import numpy as np
from scipy.linalg import inv

# Your measurement
z = np.array([x_meas, y_meas])

# Your prediction
z_pred = np.array([x_pred, y_pred])

# Innovation
y = z - z_pred

# Innovation covariance (from filter)
# S = H @ P @ H^T + R
H = np.eye(2, 9)  # Measures position
S = H @ P_matrix @ H.T + R_matrix

# NIS
nis = y.T @ inv(S) @ y

# Should be chi-squared distributed with df=2
# Mean should be 2, 95% should be < 5.99
```

---

## What Values Are "Good"?

Based on your current setup:

| Parameter              | Typical Range   | Your Values |
| ---------------------- | --------------- | ----------- |
| UWB measurement noise  | 0.05 - 0.2 m    | 0.1 m       |
| IMU attitude noise     | 0.01 - 0.05 rad | 0.02 rad    |
| Position process noise | 1e-3 to 1e-1    | 1e-2        |
| Velocity process noise | 1e-4 to 1e-2    | 1e-3        |
| Attitude process noise | 1e-3 to 1e-1    | 1e-2        |

These are starting points - use NIS test to tune for your specific system.

---

## Questions to Ask Your Data

1. **Is the filter consistent?** → NIS test
2. **Is it biased?** → Innovation mean
3. **Is it stable?** → Covariance trace
4. **Can it observe everything?** → State observability
5. **Does it trust measurements?** → Innovation std vs. R
6. **Is it numerically sound?** → Condition number, eigenvalues
