"""
This script processes the Heb_NFC legacy dataset. It serves as a demo for the minimal way (up to some
dataset-specific configurations) to run an end-to-end behavioral pipeline.

For more info on Heb_NFC, see https://comdepri.slab.com/posts/legacy-cfg-data-wehb62pp#hy4fc-heb-nfc
"""

from CFGpy.behavioral import Configuration, Pipeline

config = Configuration.default()
config.GAME_ID = "e10906ea-fc0b-458b-958c-124da688c70a"

pipeline = Pipeline(config=config)
features = pipeline.run_pipeline()
print(features)
