"""
Cofytra AI package (contract v1).

NOTE (backend import, 2026-11-09): the AI team's original ``__init__.py`` was not supplied with the
contract files. This minimal file only provides the imports documented in
docs/COFYTRA_AI_BACKEND_CONTRACT_V1.md section 2:

    from cofytra_ai import service, CofytraAIError

Replace it with the AI team's original when it is delivered.
"""

from cofytra_ai import models, service
from cofytra_ai.errors import CofytraAIError

__all__ = ["models", "service", "CofytraAIError"]
