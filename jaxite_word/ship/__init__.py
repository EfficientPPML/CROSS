"""SHIP through CROSS's context-owned bootstrap facade."""

import sys

sys.modules.setdefault("ship", sys.modules[__name__])
sys.modules.setdefault("jaxite_word.ship", sys.modules[__name__])

# Register both package spellings before importing the shared public classes.
from ship.api import SHIPBootstrap  # noqa: E402
from ship.plan import SHIPConfig  # noqa: E402

# Keep configuration class identity stable for both supported import styles.
for _name in ("api", "plan"):
  sys.modules.setdefault(f"jaxite_word.ship.{_name}", sys.modules[f"ship.{_name}"])
del _name

__all__ = ["SHIPBootstrap", "SHIPConfig"]
