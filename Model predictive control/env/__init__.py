#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:11:28 2019

@author: root
"""
from gym.envs.registration import register
register(
    id='AutoDrillEnv-v0',
    entry_point='env.AutoDrillEnv:AutoDrillEnv',
)

