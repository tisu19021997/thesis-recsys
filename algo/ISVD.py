import numpy as np

from lib.funk_svd import SVD
from lib.funk_svd.fast_methods import _run_epoch


class ISVD(SVD):
    def partial_fit(self, X):
        X = X.copy()

        u_ids = X['u_id'].unique().tolist()
        i_ids = X['i_id'].unique().tolist()

        n_user = len(self.user_dict)
        n_item = len(self.item_dict)
        n_new_user = 0
        n_new_item = 0

        for u_id in u_ids:
            if u_id not in self.user_dict:
                self.user_dict[u_id] = n_user
                n_user += 1
                n_new_user += 1

        for i_id in i_ids:
            if i_id not in self.item_dict:
                self.item_dict[i_id] = n_item
                n_item += 1
                n_new_item += 1

        X['u_id'] = X['u_id'].map(self.user_dict).astype(np.int32)
        X['i_id'] = X['i_id'].map(self.item_dict).astype(np.int32)

        # Start with the previous model
        pu = self.pu
        qi = self.qi
        bu = self.bu
        bi = self.bi

        X = X.values

        # Add rows for existed pu, qi, bu, bi
        pu_ = np.random.normal(0, .1, (n_new_user, self.n_factors))
        qi_ = np.random.normal(0, .1, (n_new_item, self.n_factors))
        bu_ = np.zeros(n_new_user)
        bi_ = np.zeros(n_new_item)

        pu = np.concatenate((pu, pu_))
        qi = np.concatenate((qi, qi_))
        bu = np.concatenate((bu, bu_))
        bi = np.concatenate((bi, bi_))

        # Run SGD for 1 epoch
        start = self._on_epoch_begin(0)
        pu, qi, bu, bi = _run_epoch(X, pu, qi, bu, bi, self.global_mean, self.n_factors, self.lr, self.reg)
        self._on_epoch_end(start)

        self.pu = pu
        self.qi = qi
        self.bu = bu
        self.bi = bi
