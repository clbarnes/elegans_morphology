from tqdm import tqdm

from morphology.common import POW, MorphologyReader

TEST_NEURON = "ASGL"  # it has dendrite

if __name__ == '__main__':
    with POW() as ow:
        asgl = ow.net.aneuron(TEST_NEURON)

    reader = MorphologyReader()
    asgl_morph = reader.get_morphology(TEST_NEURON)

    all_morphs = dict(tqdm(reader.get_all_morphologies()))
