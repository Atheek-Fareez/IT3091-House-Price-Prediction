"""IT3091 Machine Learning Group Project Source Package.
Group 2026-AI-08K · Ames Residential Property Valuation
"""
import sys

# Provide backward-compatible module aliases so notebooks referencing
# either 'src.member_03' or 'src.member_03_wazni_ahamed' work seamlessly.
try:
    from . import member_01_atheek_fareez as member_01
    sys.modules["src.member_01"] = member_01
except ImportError:
    pass

try:
    from . import member_03_wazni_ahamed as member_03
    sys.modules["src.member_03"] = member_03
except ImportError:
    pass

try:
    from . import member_04_raashidh as member_04
    sys.modules["src.member_04"] = member_04
except ImportError:
    pass
