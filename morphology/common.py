import os
from glob import iglob
from pathlib import Path
from typing import Iterator, Tuple

import neuroml
import neuroml.loaders as loaders

import PyOpenWorm as P
from PyOpenWorm.context import Context
from PyOpenWorm.worm import Worm
from PyOpenWorm.network import Network

from .constants import POW_CONF_PATH, WORM_IDENT, NML_ROOT


class POW:
    def __init__(self):
        self.ctx: Context = None
        self.worm: Worm = None
        self.net: Network = None

    def __enter__(self):
        # if P.connected:
        #     raise RuntimeError("Already connected")
        P.connect(POW_CONF_PATH)
        self.ctx = Context(ident=WORM_IDENT)
        self.worm = self.ctx.stored(Worm)()
        self.net = self.worm.neuron_network()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ctx = None
        self.worm = None
        self.net = None
        P.disconnect()


class MorphologyReader:
    def __init__(self, root_dir=NML_ROOT, cell_suffix='.cell.nml', loader=loaders.NeuroMLLoader):
        self.root_dir = Path(root_dir)
        self.cell_suffix = cell_suffix
        self.loader = loader

    def _name_to_path(self, neuron):
        if not isinstance(neuron, str):
            neuron = neuron.name()
        return self.root_dir / (neuron.upper() + self.cell_suffix)

    def load_file(self, path):
        return self.loader.load(str(path))

    def get_cell(self, path) -> neuroml.Cell:
        return self.load_file(path).cells[0]

    def get_morphology(self, name) -> neuroml.Morphology:
        return self.get_cell(self._name_to_path(name)).morphology

    def get_morphologies(self, names) -> Iterator[Tuple[str, neuroml.Morphology]]:
        for name in names:
            yield name, self.get_morphology(name)

    def get_all_morphologies(self) -> Iterator[Tuple[str, neuroml.Morphology]]:
        yield from self.get_morphologies((
            os.path.basename(f)[:-len(self.cell_suffix)] for f in iglob(str(self.root_dir / ("*" + self.cell_suffix)))
        ))


def generate_ids(i=1):
    while True:
        yield i
        i += 1


id_gen = generate_ids(1)

