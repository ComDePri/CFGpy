from CFGpy.behavioral.data_structs import PreprocessedDataset
from CFGpy.behavioral import Preprocessor
from utils import get_vanilla, get_orig_map, dump_vanilla_descriptors

# combine parsed vanilla data sets:
# TODO: combine them and call utils.dump_vanilla


# update derived data:
parsed_vanilla = get_vanilla()
preprocessed_vanilla = Preprocessor(parsed_vanilla).preprocess()
vanilla = PreprocessedDataset(preprocessed_vanilla)

step_counter, _ = vanilla.step_counter()
step_orig_map = get_orig_map(step_counter, group_func=lambda step: step[0])
gallery_counter, _ = vanilla.gallery_counter()
gallery_orig_map = get_orig_map(gallery_counter)
giant_component = vanilla._calc_giant_component()

dump_vanilla_descriptors(step_orig_map, gallery_orig_map, giant_component)
