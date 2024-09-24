# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:04:50 2019

@author: kaneko.naoshi
"""

import argparse
import glob
import math
import os
import random
import sys

import bpy
import numpy as np

sys.path.append(os.getcwd())
from config import opts


def set_random_seed(seed):
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)


def deg2rad(deg):
    return tuple(map(lambda x: math.radians(x), deg))


def random_rot(deg, rot_range=5):
    rot = random.randint(-rot_range, rot_range)
    return tuple(map(lambda x: x + rot, deg))


def random_pos(pos1, pos2):
    return tuple(map(lambda p1, p2: random.uniform(p1, p2), pos1, pos2))


def random_color():
    return tuple([random.uniform(0, 1) for i in range(3)])


def random_grayscale():
    return (random.uniform(0, 1),) * 3


def random_material(object_name):
    obj = bpy.data.objects[object_name]

    try:
        # Remove current materials
        obj.data.materials.clear()

        # Create new material
        mat = bpy.data.materials.new('Random.Mat')
        mat.diffuse_color = random_color()

        # Assign the new material to a new material slot
        bpy.context.scene.objects.active = obj  # Make the object active
        bpy.ops.object.material_slot_add()
        bpy.context.object.active_material = mat

        # Create new texture
        tex_types = ['NONE', 'BLEND', 'CLOUDS', 'DISTORTED_NOISE', 'MAGIC',
                     'MARBLE', 'MUSGRAVE', 'NOISE', 'STUCCI', 'VORONOI', 'WOOD']
        tex = bpy.data.textures.new('Random.Tex', type=random.choice(tex_types))

        # Assign the new texture to a new texture slot
        tex_slot = mat.texture_slots.add()
        tex_slot.color = random_color()
        tex_slot.texture = tex
    except AttributeError:
        pass  # Object is EMPTY type


def remove_object(object_name):
    try:
        obj = bpy.data.objects[object_name]
        bpy.data.objects.remove(obj, do_unlink=True)
    except KeyError:
        pass


def cleanup():
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)


def create_camera(name, lens, sensor_width):
    scene = bpy.context.scene

    # Create new camera datablock
    camera_data = bpy.data.cameras.new(name=name)

    # Set lens and sensor width
    camera_data.lens = lens
    camera_data.sensor_width = sensor_width

    # Create new object with the datablock
    camera_object = bpy.data.objects.new(name=name, object_data=camera_data)

    # Link the object to the scene
    scene.objects.link(camera_object)

    # Assign the object as camera
    scene.camera = camera_object

    return camera_object


def create_lamp(name, energy, distance, location):
    scene = bpy.context.scene

    # Create new lamp datablock
    lamp_data = bpy.data.lamps.new(name=name, type='POINT')

    # Set energy and distance
    lamp_data.energy = energy
    lamp_data.distance = distance

    # Enable shadow casting
    lamp_data.shadow_method = 'RAY_SHADOW'
    lamp_data.shadow_color = random_grayscale()

    # Create new object with the datablock
    lamp_object = bpy.data.objects.new(name=name, object_data=lamp_data)

    # Link the object to the scene
    scene.objects.link(lamp_object)

    # Place lamp to a specified location
    lamp_object.location = location

    return lamp_object


def load_cad_model(cad_path):
    # Import obj model
    bpy.ops.import_scene.fbx(filepath=cad_path)

    # Importer automatically selects the newly imported object
    cad = bpy.context.selected_objects[0]

    return cad


def load_food_model(food_path, scale=0.01):
    # Import obj model
    bpy.ops.import_scene.obj(filepath=food_path)

    # Importer automatically selects the newly imported object
    food = bpy.context.selected_objects[0]

    # Set scale
    food.scale[0] = food.scale[1] = food.scale[2] = scale

    # Reset rotation
    food.rotation_euler = deg2rad((0, 0, 0))

    # Move the object center to the origin
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')

    return food


def render_scene(object_name, pass_index, xres, yres,
                 mask_out, render_out, depth_out, normal_out):
    # Get scene
    scene = bpy.data.scenes['Scene']

    # Activate object index pass
    if not scene.render.layers['RenderLayer'].use_pass_object_index:
        scene.render.layers['RenderLayer'].use_pass_object_index = True
    
    # Activate normal pass
    if not scene.render.layers['RenderLayer'].use_pass_normal:
        scene.render.layers['RenderLayer'].use_pass_normal = True

    # Activate nodes
    if not bpy.context.scene.use_nodes:
        bpy.context.scene.use_nodes = True

    # Enable ray tracing to cast shadows
    if not scene.render.use_raytrace:
        scene.render.use_raytrace = True
    
    # Set rendering property
    scene.render.resolution_x = xres
    scene.render.resolution_y = yres
    scene.render.resolution_percentage = 100

    # Set a pass index to the object
    bpy.data.objects[object_name].pass_index = pass_index

    # Delete all nodes
    node_tree = bpy.context.scene.node_tree
    node_tree.nodes.clear()

    # Render Layers node
    render_layers = node_tree.nodes.new(type='CompositorNodeRLayers')

    # ID Mask node
    id_mask_node = node_tree.nodes.new(type='CompositorNodeIDMask')
    id_mask_node.index = pass_index

    # Connect object index to the node
    index_ob_output = render_layers.outputs.get('IndexOB')
    node_tree.links.new(index_ob_output, id_mask_node.inputs[0])

    # Set output node for the mask image (.png)
    mask_output_node = node_tree.nodes.new(type='CompositorNodeOutputFile')
    mask_output_node.base_path = os.path.abspath(mask_out)
    mask_output_node.file_slots[0].path = ''

    # Connect the ID Mask node to the output node
    alpha_output = id_mask_node.outputs.get('Alpha')
    node_tree.links.new(alpha_output, mask_output_node.inputs[0])

    # Set output node for a rendered image (.png)
    render_output_node = node_tree.nodes.new(type='CompositorNodeOutputFile')
    render_output_node.base_path = os.path.abspath(render_out)
    render_output_node.file_slots[0].path = ''

    # Connect the rendered image to the output node
    image_output = render_layers.outputs.get('Image')
    node_tree.links.new(image_output, render_output_node.inputs[0])

    # Set output node for a depth map (.exr)
    depth_output_node = node_tree.nodes.new(type='CompositorNodeOutputFile')
    depth_output_node.base_path = os.path.abspath(depth_out)
    depth_output_node.file_slots[0].path = ''
    depth_output_node.format.file_format = 'OPEN_EXR'

    # Connect the normal map to the output node
    depth_output = render_layers.outputs.get('Depth')
    node_tree.links.new(depth_output, depth_output_node.inputs[0])

    # Set output node for a normal map (.exr)
    normal_output_node = node_tree.nodes.new(type='CompositorNodeOutputFile')
    normal_output_node.base_path = os.path.abspath(normal_out)
    normal_output_node.file_slots[0].path = ''
    normal_output_node.format.file_format = 'OPEN_EXR'

    # Connect the normal map to the output node
    normal_output = render_layers.outputs.get('Normal')
    node_tree.links.new(normal_output, normal_output_node.inputs[0])

    # Make output directories
    if not os.path.isdir(mask_out):
        os.makedirs(mask_out)
    if not os.path.isdir(render_out):
        os.makedirs(render_out)
    if not os.path.isdir(depth_out):
        os.makedirs(depth_out)
    if not os.path.isdir(normal_out):
        os.makedirs(normal_out)

    # Render the scene
    bpy.ops.render.render(use_viewport=True)

    # Delete all nodes
    node_tree.nodes.clear()


def run_blender(food_paths, cad_path, door_path, camera_params, camera_pos,
                lamp_params, lamp_pos, tray_pos, randomized_obj, n_samples, out):
    # Remove all the objects
    cleanup()

    # Create camera
    camera = create_camera('Camera', camera_params['lens'],
                           camera_params['sensor_width'])

    # Load CAD model
    load_cad_model(cad_path)
    load_cad_model(door_path)

    # Make output directory
    if not os.path.isdir(out):
        os.makedirs(out)

    for food_path in food_paths:
        # Import food model
        food = load_food_model(food_path)

        food_dir = os.path.basename(os.path.dirname(food_path))

        # Ouput directories
        mask_out = os.path.join(out, food_dir, 'mask')
        render_out = os.path.join(out, food_dir, 'render')
        depth_out = os.path.join(out, food_dir, 'depth')
        normal_out = os.path.join(out, food_dir, 'normal')

        # Object pass index
        pass_index = 1

        # Camera parameters
        xres = camera_params['xres']
        yres = camera_params['yres']

        # Lamp parameters
        lamp_energy = lamp_params['energy']
        lamp_dist = lamp_params['distance']

        # Frame number (starts at 1)
        for frame_number in range(1, n_samples + 1):
            # Set frame number
            bpy.context.scene.frame_set(frame_number)

            # Set camera position
            camera.location = camera_pos['loc']
            camera_rot = random_rot(camera_pos['rot'])
            camera.rotation_euler = deg2rad(camera_rot)

            # Create lamp
            rand_energy = random.uniform(lamp_energy[0], lamp_energy[1])
            rand_dist = random.uniform(lamp_dist[0], lamp_dist[1])
            rand_loc = random_pos(lamp_pos[0], lamp_pos[1])
            lamp = create_lamp('Lamp', rand_energy, rand_dist, rand_loc)

            # Rotate food object horizontally
            food_rot = random.randint(0, 360)
            food.rotation_euler = deg2rad((0, 0, food_rot))

            # Place a food just above the oven's gridiron
            food.location = random_pos(tray_pos[0], tray_pos[1])
            bbox_height = food.dimensions[2]
            food.location[2] += bbox_height / 2

            # Domain randomization for the cad model
            for obj in bpy.data.objects:
                if any(s in obj.name for s in randomized_obj):
                    random_material(obj.name)

            # Scene rendering (rendered image, mask)
            render_scene(food.name, pass_index, xres, yres,
                         mask_out, render_out, depth_out, normal_out)
            
            # Remove lamp
            remove_object(lamp.name)
            
        # Remove food model
        remove_object(food.name)

    # Remove all the objects
    cleanup()
    

def main():
    argv = sys.argv

    if '--' in argv:
        argv = argv[argv.index('--') + 1:]  # get all args after '--'

    parser = argparse.ArgumentParser(
        description='Blender script to create synthetic depth estimation dataset')
    parser.add_argument('--foods', '-f', required=True,
                        help='Directory stores source food 3d models')
    parser.add_argument('--seed', '-s', type=int, default=1,
                        help='Random seed for reproducibility')
    parser.add_argument('--out', '-o', required=True,
                        help='Directory that generated dataset will be stored')
    args = parser.parse_args(argv)

    set_random_seed(args.seed)

    train_food_dir = os.path.join(args.foods, 'train')
    val_food_dir = os.path.join(args.foods, 'val')

    train_food_paths = sorted(glob.glob(os.path.join(train_food_dir, '**', '*.obj'),
                              recursive=True))
    val_food_paths = sorted(glob.glob(os.path.join(val_food_dir, '**', '*.obj'),
                            recursive=True))

    if not train_food_paths:
        raise ValueError('No food obj models are found in ' + train_food_dir)
    if not val_food_paths:
        raise ValueError('No food obj models are found in ' + val_food_dir)

    # Scene configurations
    cad_path = opts['cad_path']
    door_path = opts['door_path']
    tray_pos = opts['tray_pos']
    camera_params = opts['camera_params']
    camera_pos = opts['camera_pos']
    lamp_params = opts['lamp_params']
    lamp_pos = opts['lamp_pos']

    # Camera parameters
    xres = camera_params['xres']
    yres = camera_params['yres']

    # Domain randomization
    randomized_obj = opts['randomized_obj']

    # Number of samples
    n_train = opts['n_train']
    n_val = opts['n_val']

    # Training set
    train_out = os.path.join(args.out, 'train')
    run_blender(train_food_paths, cad_path, door_path, camera_params, camera_pos,
                lamp_params, lamp_pos, tray_pos, randomized_obj, n_train, train_out)

    # Validation set
    val_out = os.path.join(args.out, 'val')
    run_blender(val_food_paths, cad_path, door_path, camera_params, camera_pos,
                lamp_params, lamp_pos, tray_pos, randomized_obj, n_val, val_out)


if __name__ == '__main__':
    main()
