# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 19:05:56 2019

@author: kaneko.naoshi
"""

opts = {
    'cad_path': 'models/cavity.fbx',
    'door_path': 'models/door.fbx',
    'tray_pos': [(-0.04688, 0.22912, -0.26639), (0.04969, 0.19104, -0.26639)],
    'camera_params': {
        'xres': 320, 'yres': 240, 'lens': 56.0, 'sensor_width': 58.0
    },
    'camera_pos': {
        'loc': (-0.23404, 0.22858, -0.03557), 'rot': (40, 0,  270)
    },
    'lamp_params': {
        'energy': [0.01, 1.0], 'distance': [0.01, 0.5]
    },
    'lamp_pos': [
        (-0.13000, 0.22895, -0.03881), (0.10409, 0.25897, -0.03881)
    ],
    'randomized_obj': [
        'Untitled.004', 'Untitled.162', 'Untitled.163', 'Untitled.172', 'Untitled.173'
    ],
    'n_train': 300, 'n_val': 30
}
