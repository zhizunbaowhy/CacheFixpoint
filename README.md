# CacheFixpoint

Tool for analyzing cache state using abstract interpretation and fixed-point algorithms.

Basic usage:

```python
from fixpoint import *

f = r"examples/example-0.in"
config, graph, user_kwargs = read_from_file(f)
is_fixpoint = fixpoint(config, graph, 'must', **user_kwargs)  # Switch to `may` or `persistent`.
```

See ``main.ipynb`` for details.

The project is based on the MIT license.
