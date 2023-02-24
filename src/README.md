# ReadMe

## File Structure

```bash
.
├── README.md
├── __init__.py
├── chamfer_distance
│   ├── README.md
│   ├── __init__.py
│   ├── chamfer_distance.cpp
│   ├── chamfer_distance.cu
│   └── chamfer_distance.py
├── fps.py
├── quaternion.py
├── trainer.py
└── utils
    └── __init__.py
```

## Dependencies

The Chamfer Distance is downloaded from <https://github.com/chrdiller/pyTorchChamferDistance>.

The `emd_torch` is downloaded from <https://github.com/vinits5/emd>.

```bash
cd src/loss/emd_torch
python setup.py install
```
