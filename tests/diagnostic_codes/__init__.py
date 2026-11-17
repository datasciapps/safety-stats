"""One triggering case per diagnostic code, run by ``tests/test_diagnostic_codes.py``.

Each module beside ``harness.py`` holds the cases of one kind of source and exports them as
``CASES``. The modules have no ``test_`` prefix, so that pytest collects them only through
``tests/test_diagnostic_codes.py``.
"""
