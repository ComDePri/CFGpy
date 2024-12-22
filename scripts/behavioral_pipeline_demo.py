"""
This script processes the Heb_NFC legacy dataset. It serves as a demo for the minimal way (up to some
dataset-specific configurations) to run an end-to-end behavioral pipeline.

For more info on Heb_NFC, see https://comdepri.slab.com/posts/legacy-cfg-data-wehb62pp#hy4fc-heb-nfc
"""

from CFGpy.behavioral import Configuration, Pipeline

config = Configuration.default()
config.RED_METRICS_CSV_URL = r"https://api.creativeforagingtask.com/v1/event.csv?game=c9d8979c-94ad-498f-8d2b-a37cff3c5b41&gameVersion=40f2894d-1891-456b-af26-a386c6111287&entityType=event&before=2024-12-05T0:00:00.000Z"
config.PARSER_ID_COLUMNS = ("userProvidedId",)  # required for this specific dataset

pipeline = Pipeline(config=config)
features = pipeline.run_pipeline()
print(features)
