from CFGpy.behavioral.data_classes import PostparsedDataset
from CFGpy.behavioral import PostParser
from utils import get_vanilla, dump_vanilla_descriptors

# combine parsed vanilla data sets:
# TODO: combine them and call utils.dump_vanilla


# update derived data:
parsed_vanilla = get_vanilla()
postparsed_vanilla = PostParser(parsed_vanilla).postparse()
vanilla = PostparsedDataset(postparsed_vanilla)

step_counter, _ = vanilla.step_counter()
gallery_counter, _ = vanilla.gallery_counter()
giant_component = vanilla._calc_giant_component()

dump_vanilla_descriptors(step_counter, gallery_counter, giant_component)
