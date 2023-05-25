import gzip
from collections import OrderedDict
from os import PathLike
from typing import Any, Iterator
import msgpack


def load_msgpack_l_gz(filename: PathLike) -> Iterator[Any]:
    with gzip.open(filename) as f:
        unpacker = msgpack.Unpacker(f, raw=False, object_pairs_hook=OrderedDict)
        yield from unpacker
