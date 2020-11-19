#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Chordality testing"""
import unittest
import numpy as np

from scipy.sparse import csr_matrix

from sknetwork.data import house, bow_tie, karate_club
from sknetwork.data.test_graphs import test_graph_empty, test_graph_clique
from sknetwork.topology.chordality import *


class TestChordality(unittest.TestCase):

    def test_empty(self):
        adjacency = test_graph_empty()
        self.assertTrue(is_chordal(adjacency))

    def test_cliques(self):
        adjacency = test_graph_clique()
        self.assertTrue(is_chordal(adjacency))

    def test_house(self):
        adjacency = house()
        self.assertFalse(is_chordal(adjacency))

    def test_bow_tie(self):
        adjacency = bow_tie()
        self.assertTrue(is_chordal(adjacency))

    def test_karate_club(self):
        adjacency = karate_club()
        self.assertFalse(is_chordal(adjacency))

    def paper_graph_test(self):
        row = np.array([0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5])
        col = np.array([2, 3, 2, 4, 0, 1, 5, 0, 5, 1, 5, 2, 3, 4])
        data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        graph_test_1 = csr_matrix((data, (row, col)), shape=(6, 6))
        self.assertFalse(is_chordal(graph_test_1))
