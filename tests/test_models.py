"""Behavioral regression checks for the model and split-dependent training state."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logging
import unittest
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from daisy.model.SimpleXRecommender import SimpleX
from daisy.model.LightGCNRecommender import LightGCN
from daisy.utils.utils import fit_model
from daisy.utils.dataset import BasicDataset, CandidatesDataset, get_dataloader


def config():
    return dict(gpu='', logger=logging.getLogger('regression'), epochs=1, lr=.001,
                topk=2, user_num=3, item_num=4, factors=2, num_layers=1,
                reg_1=0., reg_2=0., margin=.7, negative_weight=10., gamma=.5,
                dropout=0., loss_type='BPR', optimizer='default', init_method='default',
                early_stop=False, train_ur={0:{0,1}, 1:{2}}, algo_name='simplex',
                num_workers=0, batch_size=4, num_ng=2, UID_NAME='user', IID_NAME='item',
                INTER_NAME='rating', sample_method='uniform', sample_ratio=0)


class ModelRegression(unittest.TestCase):
    def test_vector_batches_preserve_standard_loader_order(self):
        dataset = BasicDataset(np.arange(93).reshape(31,3))
        for shuffle in [False,True]:
            torch.manual_seed(2022)
            expected = list(torch.utils.data.DataLoader(dataset,batch_size=7,shuffle=shuffle))
            torch.manual_seed(2022)
            actual = list(get_dataloader(dataset,7,shuffle,0))
            self.assertEqual(len(expected),len(actual))
            for left,right in zip(expected,actual):
                for column_left,column_right in zip(left,right):
                    torch.testing.assert_close(column_left,column_right)

    def test_history_mean_projection_empty_and_gradient(self):
        model = SimpleX(config())
        with torch.no_grad():
            model.embed_user.weight.zero_()
            model.embed_item.weight.copy_(torch.tensor([[2.,0.],[0.,4.],[6.,2.],[99.,99.]]))
            model.history_projection.weight.copy_(torch.eye(2))
        representation = model.get_user_representation(torch.tensor([0,1,2,0]))
        torch.testing.assert_close(representation, torch.tensor([[.5,1.],[3.,1.],[0.,0.],[.5,1.]]))
        representation.sum().backward()
        self.assertGreater(model.embed_item.weight.grad[:3].abs().sum().item(),0)
        self.assertEqual(model.embed_item.weight.grad[3].abs().sum().item(),0)
        # Held-out item is never pulled into the history representation.
        with torch.no_grad():
            model.embed_item.weight[3].fill_(-200.)
        torch.testing.assert_close(model.get_user_representation(torch.tensor([0])), representation[:1])

    def test_training_history_is_copied_and_required(self):
        recipe = config()
        model = SimpleX(recipe)
        recipe['train_ur'][0].add(3)
        self.assertNotIn(3, model.history_items.tolist())
        recipe.pop('train_ur')
        with self.assertRaisesRegex(ValueError, 'train_ur'):
            SimpleX(recipe)

    def test_lightgcn_uses_normalized_graph_and_layer_average(self):
        recipe = config()
        recipe['user_num'], recipe['item_num'] = 2,2
        recipe['inter_matrix'] = sp.coo_matrix(([1.,1.],([0,1],[0,1])),shape=(2,2))
        model = LightGCN(recipe)
        with torch.no_grad():
            model.embed_user.weight.copy_(torch.tensor([[2.,0.],[0.,4.]]))
            model.embed_item.weight.copy_(torch.tensor([[6.,0.],[0.,8.]]))
        users, items = model.forward()
        expected = torch.tensor([[4.,0.],[0.,6.]])
        torch.testing.assert_close(users, expected)
        torch.testing.assert_close(items, expected)
        recipe['inter_matrix'] = sp.coo_matrix(([1.,1.],([0,1],[1,0])),shape=(2,2))
        changed = LightGCN(recipe)
        with torch.no_grad():
            changed.embed_user.weight.copy_(model.embed_user.weight)
            changed.embed_item.weight.copy_(model.embed_item.weight)
        self.assertFalse(torch.allclose(changed.forward()[0], users))

    def test_both_models_build_and_rank_single_user_from_training_split(self):
        train = pd.DataFrame({'user':[0,0,1], 'item':[0,1,2], 'rating':[1.,1.,1.]})
        for algorithm in ['simplex','lightgcn']:
            recipe = config()
            recipe['algo_name'] = algorithm
            model = fit_model(train.copy(), recipe)
            loader = get_dataloader(CandidatesDataset([[2,np.array([0,1,2,3])]]),1,False,0)
            with torch.no_grad():
                result = model.rank(loader)
            self.assertEqual(result.shape,(1,2))
            self.assertEqual(result.dtype.kind,'i')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
