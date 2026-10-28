"""One exploratory training birth with a selectable reward profile (Claude, issue #19).

Thin entry point; see ``noesis_gpu.profile_run`` for behavior and exit codes.
"""
import sys

from noesis_gpu.profile_run import main

sys.exit(main())
