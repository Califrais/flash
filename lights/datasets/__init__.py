"""
This modules includes dataset loaders for experiments conducted with WildWood
"""

from .dataset import Dataset

from .loader import (
    load_PBC_Seq,
)


loader_from_name = {
    "PBC_Seq": load_PBC_Seq,
}
