import numpy as np

from surprise import SVD
from surprise.utils import get_rng


class IncrementalSVD(SVD):
    def fold_in(self, new_ratings, verbose=True):
        for uid, iid, rating in new_ratings:
            try:
                inner_uid = self.trainset.to_inner_uid(uid)
            except (ValueError, KeyError) as e:
                # if new user, increase total number of user by 1
                # append new row to the `raw2inner_id_users` matrix
                inner_uid = self.trainset.n_users
                self.trainset.n_users += 1
                self.trainset._raw2inner_id_users[uid] = inner_uid

                if self.trainset._inner2raw_id_users is not None:
                    self.trainset._inner2raw_id_users[inner_uid] = uid

            try:
                inner_iid = self.trainset.to_inner_iid(iid)
            except (ValueError, KeyError) as e:
                # if new item, same as new user.
                inner_iid = self.trainset.n_items
                self.trainset.n_items += 1
                self.trainset._raw2inner_id_items[iid] = inner_iid

                if self.trainset._inner2raw_id_items is not None:
                    self.trainset._inner2raw_id_items[inner_iid] = iid

            # append new item with rating in the user profile
            self.trainset.ur[inner_uid].append((inner_iid, rating))
            # append new user with rating in item profile
            self.trainset.ir[inner_iid].append((inner_uid, rating))

        new_ratings_vec = [(self.trainset.to_inner_uid(uid), self.trainset.to_inner_iid(iid), rating) for
                           uid, iid, rating in new_ratings]
        self.partial_fit(new_ratings_vec, random_state=self.random_state, verbose=verbose)

    def partial_fit(self, new_ratings, random_state=None, verbose=True):
        rng = get_rng(random_state=random_state)

        # start with the last trained model
        bu = self.bu
        bi = self.bi
        pu = self.pu
        qi = self.qi

        if not self.biased:
            global_mean = 0
        else:
            global_mean = self.trainset.global_mean

        # for current_epoch in range(self.n_epochs):
        # only run for 1 epoch?
        for u, i, r in new_ratings:
            # compute current error
            dot = 0  # <q_i, p_u>

            # if user is new, append new row to `pu`, new column to `bu`
            if u > len(pu) - 1:
                pu = np.concatenate((pu, rng.normal(self.init_mean, self.init_std_dev, (1, self.n_factors))),
                                    axis=0)
                bu = np.append(bu, 0)

                # same for item
            if i > len(qi) - 1:
                qi = np.concatenate((qi, rng.normal(self.init_mean, self.init_std_dev, (1, self.n_factors))),
                                    axis=0)
                bi = np.append(bi, 0)

            for f in range(self.n_factors):
                dot += qi[i, f] * pu[u, f]

            # compute the error
            err = r - (global_mean + bu[u] + bi[i] + dot)

            # if verbose:
            #     # sys.stdout.write('\r')
            #     print(f'Epoch {current_epoch + 1}/{self.n_epochs}: loss: {abs(err)}')  # , end='')

            # update biases
            if self.biased:
                bu[u] += self.lr_bu * (err - self.reg_bu * bu[u])
                bi[i] += self.lr_bi * (err - self.reg_bi * bi[i])

            # update factors
            for f in range(self.n_factors):
                puf = pu[u, f]
                qif = qi[i, f]
                pu[u, f] += self.lr_pu * (err * qif - self.reg_pu * puf)
                qi[i, f] += self.lr_qi * (err * puf - self.reg_qi * qif)

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi

        return self
