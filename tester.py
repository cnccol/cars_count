import numpy as np
import pandas as pd
import counter

test_expected_results = pd.read_csv(".csv")

FRAMES = list(test_expected_results['frame_number'])
VEHICLE_TYPES = list(test_expected_results['vehicle_type'])
# vehicle_number = test_expected_results['vehicle_number']
CARS = 0

for x in list(VEHICLE_TYPES):
    if x == 'car':
        CARS += 1

"""
    test a funciton name get_cars in the modules counter, the results show how good the function
    behaves in a small data.
"""
def test():
    counted_cars = 0
    frames_with_false_positives = []
    frames_with_false_negatives = []
    results = counter.get_cars()
    for i in range(len(results)):
        if results[i]['vehicle_type'] == 'car':
            if VEHICLE_TYPES[i] == 'car':
                counted_cars += 1
            else:
                frames_with_false_positives.append(FRAMES[i])
        else:
            if VEHICLE_TYPES[i] == 'car':
                frames_with_false_negatives.append(FRAMES[i])
    
    print("---------------------- Results ----------------------")
    print("Accuracy: {}".format(float(counted_cars)/float(CARS)))
    print("-----------------------------------------------------")
    print("False positives in: {}".format(frames_with_false_positives))
    print("-----------------------------------------------------")
    print("False negatives in: {}".format(frames_with_false_negatives))

