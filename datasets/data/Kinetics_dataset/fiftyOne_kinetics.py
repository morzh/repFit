import fiftyone as fo
import fiftyone.zoo as foz

kinetics_dataset_folder = '/media/anton/4c95a564-35ea-40b5-b747-58d854a622d0/home/anton/work/fitMate/dataset_aggregation/Kinetics'

dataset = foz.load_zoo_dataset(
    "kinetics-700-2020",
    dataset_dir=kinetics_dataset_folder,
    split="experiments",
    classes=["bench pressing"],
    max_samples=1000000000,
)

session = fo.launch_app(dataset)
