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

## Requirements

This program was written using Python version 3.11.2, and requires a **minimum version of 3.9** to run. It has no dependencies on external libraries.

## Reference

- Ferdinand, Christian. "**A fast and efficient cache persistence analysis.**" (1997).
- Ferdinand, Christian, Florian Martin, and Reinhard Wilhelm. "**Applying compiler techniques to cache behavior prediction.**" Proceedings of the ACM SIGPLAN Workshop on Language, Compiler and Tool Support for Real-Time Systems. Vol. 4. 1997.
- Ballabriga, Clément, and Hugues Cassé. "**Improving the first-miss computation in set-associative instruction caches.**" 2008 Euromicro Conference on Real-Time Systems. IEEE, 2008.

