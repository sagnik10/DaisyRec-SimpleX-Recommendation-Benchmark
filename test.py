"""Final testing from an explicit recipe, with results and reproducibility metadata."""
import time
from pathlib import Path
from logging import getLogger
import torch
from daisy.utils.splitter import TestSplitter
from daisy.utils.metrics import calc_ranking_results
from daisy.utils.loader import RawDataReader, Preprocessor
from daisy.utils.config import init_seed, init_config, init_logger
from daisy.utils.dataset import get_dataloader, CandidatesDataset
from daisy.utils.utils import (get_ur, build_candidates_set, fit_model, write_json,
                               serializable_config, experiment_provenance)


def main():
    config = init_config()
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    config['logger'] = logger
    provenance = experiment_provenance(config)
    processor = Preprocessor(config)
    frame = processor.process(RawDataReader(config).get_data())
    config['user_num'], config['item_num'] = processor.user_num, processor.item_num
    train_indices, test_indices = TestSplitter(config).split(frame)
    train, test = frame.iloc[train_indices].copy(), frame.iloc[test_indices].copy()
    started = time.perf_counter()
    model = fit_model(train, config)
    elapsed = time.perf_counter() - started
    test_ur = get_ur(test)
    init_seed(config['seed'], config['reproducibility'])
    users, candidates = build_candidates_set(test_ur, config['train_ur'], config)
    loader = get_dataloader(CandidatesDataset(candidates), 64, False, 0)
    with torch.no_grad():
        predictions = model.rank(loader)
    result_path = Path(config['run_dir']) / 'results' / config['dataset']
    result_path.mkdir(parents=True, exist_ok=True)
    config['res_path'] = str(result_path)
    results = calc_ranking_results(test_ur, predictions, users, config)
    results.to_csv(result_path / f'{config["algo_name"]}_kpi_results.csv', index=False)
    write_json(result_path / f'{config["algo_name"]}_manifest.json', {
        'config': serializable_config(config), 'provenance': provenance,
        'users': processor.user_num, 'items': processor.item_num,
        'train_interactions': len(train), 'test_interactions': len(test),
        'evaluated_users': len(users), 'training_seconds': elapsed,
        'actual_loss': 'CCL' if config['algo_name'] == 'simplex' else config['loss_type'],
        'status': 'complete'})
    logger.info(f'Results saved to {result_path}; training seconds={elapsed:.3f}')


if __name__ == '__main__':
    main()
