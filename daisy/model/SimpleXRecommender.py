"""SimpleX mean behavior aggregator and cosine contrastive loss (Mao et al., 2021)."""
import torch
from torch import nn
from torch.nn import functional as F
from daisy.model.AbstractRecommender import GeneralRecommender


class SimpleX(GeneralRecommender):
    tunable_param_names = ['num_ng', 'factors', 'batch_size', 'lr', 'reg_1',
                           'reg_2', 'margin', 'negative_weight', 'dropout', 'gamma']

    def __init__(self, config):
        super().__init__(config)
        for name in ['epochs', 'lr', 'topk', 'user_num', 'item_num', 'factors',
                     'reg_1', 'reg_2', 'margin', 'negative_weight', 'gamma']:
            setattr(self, name, config[name])
        if not 0 <= self.gamma <= 1:
            raise ValueError('gamma must be between 0 and 1')
        self.embed_user = nn.Embedding(self.user_num, self.factors)
        self.embed_item = nn.Embedding(self.item_num, self.factors)
        self.history_projection = nn.Linear(self.factors, self.factors, bias=False)
        self.user_dropout = nn.Dropout(config['dropout'])
        # BPR selects DaisyRec's pairwise sampler; SimpleX computes CCL below.
        if config['loss_type'].upper() not in ['BPR', 'TL', 'HL']:
            raise ValueError('SimpleX requires pairwise negative samples')
        self.loss_type = config['loss_type']
        self.optimizer = 'adam' if config['optimizer'] == 'default' else config['optimizer']
        self.initializer = 'xavier_uniform' if config['init_method'] == 'default' else config['init_method']
        self.early_stop = config['early_stop']
        self.register_buffer('history_items', torch.empty(0, dtype=torch.long))
        self.register_buffer('history_offsets', torch.zeros(self.user_num + 1, dtype=torch.long))
        self.apply(self._init_weight)
        if 'train_ur' not in config:
            raise ValueError('SimpleX requires train_ur from the current training split')
        self.set_user_history(config['train_ur'])

    def set_user_history(self, train_ur):
        """Copy only the caller's training split into a compact history."""
        items, offsets = [], [0]
        for user in range(self.user_num):
            history = sorted(set(train_ur.get(user, ())))
            if any(item < 0 or item >= self.item_num for item in history):
                raise ValueError('History contains an invalid item ID')
            items.extend(history)
            offsets.append(len(items))
        device = self.embed_item.weight.device
        self.history_items = torch.tensor(items, dtype=torch.long, device=device)
        self.history_offsets = torch.tensor(offsets, dtype=torch.long, device=device)

    def get_user_representation(self, users):
        """g * user + (1-g) * W_v(mean(training-history item embeddings))."""
        unique, inverse = torch.unique(users, return_inverse=True)
        starts = self.history_offsets[unique]
        lengths = self.history_offsets[unique + 1] - starts
        offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        positions = (torch.repeat_interleave(starts - offsets[:-1], lengths)
                     + torch.arange(offsets[-1], device=users.device))
        mean_history = F.embedding_bag(
            self.history_items[positions], self.embed_item.weight, offsets,
            mode='mean', include_last_offset=True)
        aggregate = self.history_projection(mean_history)
        representation = self.gamma * self.embed_user(unique) + (1 - self.gamma) * aggregate
        return self.user_dropout(representation[inverse])

    def forward(self, users, pos_items, neg_items=None):
        user_rep = self.get_user_representation(users)
        if neg_items is None:
            return user_rep, self.embed_item(pos_items)
        return user_rep, self.embed_item(pos_items), self.embed_item(neg_items)

    def calc_loss(self, batch):
        users, pos_items, neg_items = [column.to(self.device).long() for column in batch]
        user_rep, pos_e, neg_e = self.forward(users, pos_items, neg_items)
        user_norm = F.normalize(user_rep, dim=-1)
        pos_score = (user_norm * F.normalize(pos_e, dim=-1)).sum(-1)
        neg_score = (user_norm * F.normalize(neg_e, dim=-1)).sum(-1)
        # Mean over pairs equals mean negative penalty per positive for fixed num_ng.
        loss = (1 - pos_score + self.negative_weight * F.relu(neg_score - self.margin)).mean()
        loss = loss + self.reg_1 * (user_rep.norm(p=1) + pos_e.norm(p=1) + neg_e.norm(p=1))
        return loss + self.reg_2 * (user_rep.norm() + pos_e.norm() + neg_e.norm())

    @torch.no_grad()
    def predict(self, user, item):
        self.eval()
        users = torch.tensor([user], dtype=torch.long, device=self.device)
        representation = F.normalize(self.get_user_representation(users), dim=-1)
        item_e = F.normalize(self.embed_item.weight[item], dim=-1)
        return (representation @ item_e).item()

    @torch.no_grad()
    def rank(self, test_loader):
        self.eval()
        item_embeddings = F.normalize(self.embed_item.weight, dim=-1)
        recommendations = []
        for users, candidates in test_loader:
            users, candidates = users.to(self.device), candidates.to(self.device)
            user_e = F.normalize(self.get_user_representation(users), dim=-1)
            scores = torch.bmm(item_embeddings[candidates], user_e.unsqueeze(-1)).squeeze(-1)
            order = torch.argsort(scores, dim=1, descending=True)[:, :self.topk]
            recommendations.append(torch.gather(candidates, 1, order).cpu())
        return torch.cat(recommendations).numpy()

    @torch.no_grad()
    def full_rank(self, user):
        self.eval()
        users = torch.tensor([user], dtype=torch.long, device=self.device)
        user_e = F.normalize(self.get_user_representation(users), dim=-1)
        scores = user_e @ F.normalize(self.embed_item.weight, dim=-1).t()
        return scores.argsort(dim=1, descending=True)[0, :self.topk].cpu().numpy()
