import gzip
import random
from collections import OrderedDict
from os import PathLike
from typing import Any, Iterator, Optional

import msgpack
from dpu_utils.utils import RichPath


def load_msgpack_l_gz(filename: PathLike) -> Iterator[Any]:
    with gzip.open(filename) as f:
        unpacker = msgpack.Unpacker(f, raw=False, object_pairs_hook=OrderedDict)
        yield from unpacker
