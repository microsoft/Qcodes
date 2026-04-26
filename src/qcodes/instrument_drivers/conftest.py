import os

# Drivers that require optional third-party libraries not generally available.
# These must be excluded from doctest collection to avoid ImportErrors.
collect_ignore_glob = [
    os.path.join("Galil", "*"),
    os.path.join("Minicircuits", "*"),
    os.path.join("QuantumDesign", "DynaCoolPPMS", "private", "*"),
]
