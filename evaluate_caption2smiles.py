import argparse

from os.path import join
from evaluation import fingerprint_metrics, mol_translation_metrics, fcd_metric

MODEL_TYPES = ['t5-v1_1', 'molt5']
MODEL_SIZES = ['small', 'base', 'large']
TASK = 'caption2smiles'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='predictions/')
    parser.add_argument('--log_fp', type=str, default='caption2smiles_eval.txt')
    args = parser.parse_args()

    log_io = open(args.log_fp, 'w+')
    for model_size in MODEL_SIZES:
        for model_type in MODEL_TYPES:
            file_name = f'{model_type}-{model_size}-{TASK}.txt'
            file_path = join('predictions', file_name)
            bleu_score, exact_match_score, levenshtein_score = mol_translation_metrics.evaluate(file_path)
            validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score = fingerprint_metrics.evaluate(file_path, 2)
            fcd_metric_score = fcd_metric.evaluate(file_path)
            log_io.write(f'For {file_name}\n')
            log_io.write(f'BLEU: {round(bleu_score, 3)}\n')
            log_io.write(f'Exact: {round(exact_match_score, 3)}\n')
            log_io.write(f'Levenshtein: {round(levenshtein_score, 3)}\n')
            log_io.write(f'MACCS FTS: {round(maccs_sims_score, 3)}\n')
            log_io.write(f'RDK FTS: {round(rdk_sims_score, 3)}\n')
            log_io.write(f'Morgan FTS: {round(morgan_sims_score, 3)}\n')
            log_io.write(f'FCD Metric: {round(fcd_metric_score, 3)}\n')
            log_io.write(f'Validity: {round(validity_score, 3)}\n')
            log_io.write('\n')
            log_io.flush()
    log_io.close()
