#!/usr/bin/env python3
"""Report a config's op coverage (§spec:op-coverage, §road:coverage-report).

The measurement lives in ``ocio2onnx.census``, beside the compiler whose
supported set it reads, and is also reachable as ``ocio2onnx census``. This
script stays because SPEC.md names it by path. Requires the package
installed. Usage::

    python tools/census.py [config-uri]
"""

import sys

from ocio2onnx.census import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
