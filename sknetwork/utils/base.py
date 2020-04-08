#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 28, 2019
@author: Quentin Lutz <qlutz@enst.fr>
@author: Nathan de Lara <ndelara@enst.fr>
"""
import inspect

from scipy import sparse


class Algorithm:
    """Base class for all algorithms."""
    def __repr__(self):
        # parameters not to display
        arg_black_list = ['self', 'random_state', 'verbose']
        output = self.__class__.__name__ + '('
        signature = inspect.signature(self.__class__.__init__)
        arguments = [arg.name for arg in signature.parameters.values() if arg.name not in arg_black_list]
        for p in arguments:
            try:
                val = self.__dict__[p]
            except KeyError:
                continue
            if type(val) == str:
                val = "'" + val + "'"
            else:
                val = str(val)
            output += p + '=' + val + ', '

        if arguments:
            return output[:-2] + ')'
        else:
            return output + ')'

    def fit(self, *args, **kwargs):
        """Fit Algorithm to the data."""
        raise NotImplementedError


def bialgorithm(algorithm: Algorithm):
    """Constructor for naive algorithms on bigraphs."""
    class BiAlgo(algorithm):
        """Naive bialgorithm"""
        def __init__(self, *args, **kwargs):
            super(BiAlgo, self).__init__(*args, **kwargs)

            if hasattr(self, 'dendrogram_'):
                self.dendrogram_row_ = None
                self.dendrogram_col_ = None

            if hasattr(self, 'embedding_'):
                self.embedding_row_ = None
                self.embedding_col_ = None

            if hasattr(self, 'labels_'):
                self.labels_row_ = None
                self.labels_col_ = None

            if hasattr(self, 'membership_'):
                self.membership_row_ = None
                self.membership_col_ = None

            if hasattr(self, 'scores_'):
                self.scores_row_ = None
                self.scores_col_ = None

        def fit(self, *args, **kwargs):
            """Fit Algorithm to the data."""
            biadjacency = args[0]
            n_row, n_col = biadjacency.shape

            adjacency = sparse.bmat([[None, biadjacency], [biadjacency.T, None]], format='csr')
            new_args = list(args)
            new_args[0] = adjacency
            super(BiAlgo, self).fit(*new_args, **kwargs)

            if hasattr(self, 'dendrogram_'):
                self.dendrogram_row_ = self.dendrogram_[:n_row]
                self.dendrogram_col_ = self.dendrogram_[n_row:]
                self.dendrogram_ = self.dendrogram_row_

            if hasattr(self, 'embedding_'):
                self.embedding_row_ = self.embedding_[:n_row]
                self.embedding_col_ = self.embedding_[n_row:]
                self.embedding_ = self.embedding_row_

            if hasattr(self, 'labels_'):
                self.labels_row_ = self.labels_[:n_row]
                self.labels_col_ = self.labels_[n_row:]
                self.labels_ = self.labels_row_

            if hasattr(self, 'membership_'):
                self.membership_row_ = self.membership_[:n_row]
                self.membership_col_ = self.membership_[n_row:]
                self.membership_ = self.membership_row_

            if hasattr(self, 'scores_'):
                self.scores_row_ = self.scores_[:n_row]
                self.scores_col_ = self.scores_[n_row:]
                self.scores_ = self.scores_row_

            return self

    return BiAlgo
