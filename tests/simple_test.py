#!/usr/bin/env python3
#author markpurcell@ie.ibm.com

import logging
import unittest
import MMLL.Common_to_all_objects as utils


class SimpleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    #@unittest.skip("temporarily skipping")
    def test_mmll(self):
        utils.Common_to_all_objects()
