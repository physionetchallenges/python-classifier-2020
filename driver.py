import numpy as np
import os
import sys
from scipy.io import loadmat
from run_12ECG_classifier import load_12ECG_model, run_12ECG_classifier


def load_challenge_data(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file, 'r') as f:
        header_data = f.readlines()

    return data, header_data


def save_challenge_predictions(output_directory, filename, scores, labels, classes):

    recording = os.path.splitext(filename)[0]
    new_file = filename.replace('.mat', '.csv')
    output_file = os.path.join(output_directory, new_file)

    labels=np.asarray(labels,dtype=np.int)
    scores=np.asarray(scores,dtype=np.float64)

    # Include the filename as the recording number
    recording_string = '#{}'.format(recording)
    class_string = ','.join(classes)
    label_string = ','.join(str(i) for i in labels)
    score_string = ','.join(str(i) for i in scores)

    with open(output_file, 'w') as f:
        f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')


# Find unique number of classes
def get_classes(input_directory, files):

    classes = set()
    for f in files:
        g = f.replace('.mat', '.hea')
        input_file = os.path.join(input_directory, g)
        with open(input_file, 'r') as f:
            for lines in f:
                if lines.startswith('#Dx'):
                    tmp = lines.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())

    return sorted(classes)


if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 3:
        raise Exception('Include the input and output directories as arguments, e.g., python driver.py input output.')

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    # Find files.
    input_files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
            input_files.append(f)

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    classes = get_classes(input_directory, input_files)

    # Load model.
    print('Loading 12ECG model...')
    model = load_12ECG_model()

    # Iterate over files.
    print('Extracting 12ECG features...')
    num_files = len(input_files)

    for i, f in enumerate(input_files):
        print('    {}/{}...'.format(i+1, num_files))
        tmp_input_file = os.path.join(input_directory, f)
        data, header_data = load_challenge_data(tmp_input_file)
        current_label, current_score = run_12ECG_classifier(data, header_data, classes, model)
        # Save results.
        
        save_challenge_predictions(output_directory, f, current_score, current_label, classes)

    print('Done.')
