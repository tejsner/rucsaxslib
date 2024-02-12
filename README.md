# rucsaxslib

## Installation

Currently resides at <https://test.pypi.org/project/rucsaxslib/>.

```
pip install numpy matplotlib fabio
pip install --index-url https://test.pypi.org/simple/ --no-deps rucsaxslib
```

## Example usage

Plot detector image in laboratory coordinates (distance from Point of Normal Incidence).

```python
import rucsaxslib as rs
img = rs.from_rucsaxs('data.edf')
img.plot()
plt.show()
```

Plot detector image in polar coordinates. Rebinning is needed to avoid artifacts from `pcolormesh`.

```python
import rucsaxslib as rs
img = rs.from_rucsaxs('data.edf')
img.plot(coords='polar', rebin=300) # 300 bins in both angular directions
plt.show()
```
