import os
import pandas
import pandas as pd

from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
from tabulate import tabulate

colorama_init()

kinetics_dataset_root_folder = './csv'
extracted_classes_folder = 'extracted_classes'

train_filename = 'kinetics-600_train.csv'
validation_filename = 'kinetics-600_val.csv'
test_filename = 'kinetics-600_test.csv'

train_data = pandas.read_csv(os.path.join(kinetics_dataset_root_folder, train_filename))
validation_data = pandas.read_csv(os.path.join(kinetics_dataset_root_folder, validation_filename))
# test_data = pandas.read_csv(os.path.join(kinetics_dataset_root_folder, test_filename))

workout_classes = [
    'bench pressing',
    'deadlifting',
    'exercising with an exercise ball',
    'high jump',
    'high kick',
    'pull ups',
    'push up',
    'snatch weight lifting',
    'squat',
    'bending back',
    'situp'
]

extracted_classes_path = os.path.join(kinetics_dataset_root_folder, extracted_classes_folder)
class_videos_list = []
total_videos_number = 0

for workout_class in workout_classes:
    train_current_class = train_data.loc[train_data['label'] == workout_class]
    validation_current_class = validation_data.loc[validation_data['label'] == workout_class]
    current_class_frame = pd.concat([validation_current_class, train_current_class])
    current_class_frame.to_csv(os.path.join(extracted_classes_path, workout_class + '.csv'), index=False)
    current_number_videos = len(validation_current_class) + len(train_current_class)
    total_videos_number += current_number_videos
    # print(f'Class {Fore.LIGHTCYAN_EX + workout_class + Fore.RESET}, '
    #       f'number videos {Fore.LIGHTGREEN_EX + str(current_number_videos) + Fore.RESET}')

    class_videos_list.append([workout_class, current_number_videos])

class_videos_table = tabulate(class_videos_list, headers=['Class label', 'Videos number'])
total_classes_videos_table = tabulate([[len(workout_classes), total_videos_number]],
                                      headers=['Total Class labels      ', 'Total videos number'])

print(class_videos_table)
print('')
print(total_classes_videos_table)
