"""This file provides functions for merging evaluation files in a directory
"""
import os
import joblib
import numpy as np

###insert names from exp dir
###change .npy ~z. 47 if pretrain convert!!
EXP_NAMES = [
    #'03_Optimal_150_train_g_proposed_d_proposed_r_proposed_round',
    #'04_sub_Optimal_84_200_train_g_proposed_d_proposed_r_proposed_round',
    #'05_Notfall_140_train_g_proposed_d_proposed_r_proposed_bernoulli',
    #'05_Notfall_140_train_g_proposed_d_proposed_r_proposed_round',
    #'test_lastfm_alternative_train_g_proposed_d_proposed_r_proposed_bernoulli',
    '07_Control_test_140_125_train_g_proposed_d_proposed_r_proposed_bernoulli',
    '07_Control_test_140_125_train_g_proposed_d_proposed_r_proposed_round',

    #'03_Optimal_150_pretrain_g_proposed_d_proposed',
    #'04_sub_Optimal_84_200_pretrain_g_proposed_d_proposed',
    #'test_lastfm_alternative_pretrain_g_proposed_d_proposed',
    #'05_Notfall_140_pretrain_g_proposed_d_proposed',
    #'07_Control_test_140_125_pretrain_g_proposed_d_proposed',

]

SRC_DIRS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                 'exp', exp_name, 'eval')

    for exp_name in EXP_NAMES
]

DST_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'analysis', 'data',
                 'eval_training_progress', exp_name + '.npz')
    for exp_name in EXP_NAMES
]





def get_npy_files(target_dir):
    """Return a list of paths to all the .npy files in a directory."""
    filepaths = []
    for path in os.listdir(target_dir):
        if path.endswith('.npy'):  # .npy, round.npy, bernoulli.npy
            filepaths.append(path)
    return filepaths

def load(filepath, eval_dir):
    """Load a evaluation file at the given path and return the stored data."""
    if isinstance(os.path.splitext(filepath)[0], int):
        step = int(os.path.splitext(filepath)[0])
    else:
        step = int(os.path.splitext(filepath)[0].split('_')[0])

    data = np.load(os.path.join(eval_dir, filepath), allow_pickle=True)
    return (step, data[()]['score_matrix_mean'],
            data[()]['score_pair_matrix_mean'])

def main():
    """Main function"""
    for idx, eval_dir in enumerate(SRC_DIRS):
        filepaths = get_npy_files(eval_dir)
        collected = joblib.Parallel(n_jobs=30, verbose=5)(
            joblib.delayed(load)(filepath, eval_dir) for filepath in filepaths)

        steps = []
        score_matrix_means = []
        score_pair_matrix_means = []

        for item in collected:
            steps.append(item[0])
            score_matrix_means.append(item[1])
            score_pair_matrix_means.append(item[2])

        steps = np.array(steps)
        score_matrix_means = np.stack(score_matrix_means)
        score_pair_matrix_means = np.stack(score_pair_matrix_means)

        argsort = steps.argsort()
        steps = steps[argsort]
        score_matrix_means = score_matrix_means[argsort]
        score_pair_matrix_means = score_pair_matrix_means[argsort]

        np.savez(DST_PATHS[idx], steps=steps,
                 score_matrix_means=score_matrix_means,
                 score_pair_matrix_means=score_pair_matrix_means)

if __name__ == "__main__":
    main()
