import blenderproc as bproc
import numpy as np
# マス
import math
# ファイル名リネーム
import glob
import os
import shutil
import random
#import bpy

# 初期化
bproc.init()

obj_name = "banana.obj"
fr = 10
mat_num = 10
formats = 'colors','segmaps'

# フォルダー初期化
for format in formats:
    if(os.path.exists(format)):
        shutil.rmtree(format)

# サンプリング数調整
bproc.renderer.set_max_amount_of_samples(1)

# オブジェクト
obj = bproc.loader.load_obj("obj/"+obj_name)[0]
obj.set_cp("category_id", 1)

for n in range(fr * mat_num):
    obj.set_scale([random.uniform(0.9, 1.3),random.uniform(0.9, 1.3),random.uniform(0.9, 1.3)], frame = n)
    obj.set_rotation_euler([np.pi * 1 / 2, 0, random.uniform(0, np.pi * 2)], frame = n)
    obj.set_location([random.uniform(-1.5, 1.5),random.uniform(-6, 2),0], frame = n)

stages = bproc.loader.load_obj("obj/box.obj")

stage = {}
for n in range(len(stages)):
    stage[n] = bproc.filter.one_by_attr(stages, "name", f"stage_{n}")
    # stage[n].set_cp("category_id", n+1)
    stage[n].set_cp("category_id", 0)

# マテリアル
mat = []
for n in range(mat_num):
    mat.append(bproc.material.create_material_from_texture( f"tex/{n}.png", "mat"))

# ライト
keylight = bproc.types.Light()
filllight = bproc.types.Light()
for n in range(fr * mat_num):
    keylight.set_location([random.uniform(-10, 10), random.uniform(-10, -7), random.uniform(1, 8)], frame = n)
    filllight.set_location([random.uniform(-5, 5), random.uniform(-10, -7), random.uniform(2, 5)], frame = n)
    keylight.set_energy(random.uniform(1000, 10000), frame = n)
    filllight.set_energy(1000, frame = n)

# 回転撮影
# for n in range(fr):
#     cam_pose = bproc.math.build_transformation_mat([-5 * math.cos(math.radians(90 - 10 * n)), -5 * math.sin(math.radians(90 - 10 * n)), 0], [np.pi / 2, 0, math.radians(0 - 10 * n)])
#     bproc.camera.add_camera_pose(cam_pose)

# カメラ
cam_pose = bproc.math.build_transformation_mat([-0.048273, -13.4497, 5.85555], [1.115327, 0, -0.008726])
# bproc.camera.set_intrinsics_from_blender_params(image_width = 640, image_height = 480)
# bproc.camera.set_intrinsics_from_blender_params(image_width = 256, image_height = 192)
bproc.camera.set_intrinsics_from_blender_params(image_width = 320, image_height = 240)

# デノイズ "INTEL" or None
# bproc.renderer.set_denoiser("INTEL")

# ディスタンス・デプス・ノーマル表示切り替え
# bproc.renderer.enable_distance_output(1)
bproc.renderer.enable_depth_output(1)
# bproc.renderer.enable_depth_output(activate_antialiasing=True, output_dir="depth", file_prefix="")
# bproc.renderer.enable_normals_output(output_dir="normal", file_prefix="")
# bproc.renderer.enable_normals_output()

# テクスチャ読み込
# image = bpy.data.images.load(filepath="tex.png")

# 新規マテリアル
# blenderproc.material.create(name)

# フォーマット
# bproc.renderer.set_output_format(file_format="PNG")

for n in range (mat_num):
    # mat[n].infuse_material(mat[random.randrange(mat_num)], mode='mix', mix_strength=random.random())
    bproc.material.create_procedural_texture(pattern_name=None)
    for m in range(len(stages)):
        stage[m].set_material(0, mat[n])
    bproc.python.utility.Utility.set_keyframe_render_interval(frame_start = n*fr)
    for o in range(fr):
        cam_pose = bproc.math.build_transformation_mat([-0.048273+random.uniform(-0.2, 0.2), -13.4497+random.uniform(-0.2, 0.2), 5.85555+random.uniform(-0.2, 0.2)], [1.115327+random.uniform(-0.02, 0.02), 0, -0.008726+random.uniform(-0.02, 0.02)])
        bproc.camera.add_camera_pose(cam_pose)
    # bproc.camera.add_camera_pose(cam_pose, n*fr + fr - 1)
    # RGBレンダリング
    bproc.renderer.enable_normals_output()
    data = bproc.renderer.render()
    bproc.writer.write_hdf5("colors/", data)
    # セマンティックセグメンテーションレンダリング
    data = bproc.renderer.render_segmap(map_by=["class"])
    bproc.writer.write_hdf5("segmaps/", data)

# リネーム
# for format in formats:
#     path = './' + format
#     files = glob.glob(path + '/*')
#     for f in files:
#         n = int(os.path.splitext(os.path.basename(f))[0])
#         # os.rename(f, os.path.join(path, '{0:03d}'.format(n))+'.hdf5')
#         os.rename(f, os.path.join(path, '{0:04d}'.format(n+1))+'.hdf5')
import h5py

if(os.path.exists(obj_name.split(".")[0])):
    shutil.rmtree(obj_name.split(".")[0])

os.mkdir(obj_name.split(".")[0])

for n in range(fr*mat_num):
        with h5py.File(f"colors/{n}.hdf5", 'r') as f:
            rgb = np.array(f["colors"])
            depth = np.array(f["depth"], dtype=np.float32)
            normal = np.array(f["normals"])
            # トランスポーズ
            rgb = rgb.transpose(2, 0, 1)
            normal = normal.transpose(2, 0, 1)
            # 変換
            depth = depth - depth.min()
            depth = depth / (2 * depth.max())
        with h5py.File(f"segmaps/{n}.hdf5", 'r') as f:
            mask = np.array(f["class_segmaps"], dtype=np.bool8)
        with h5py.File(obj_name.split(".")[0]+f"/{'{0:04d}'.format(n+1)}.h5", 'w') as f:
            f.create_dataset('rgb', data=rgb)
            f.create_dataset('depth', data=depth)
            f.create_dataset('normal', data=normal)
            f.create_dataset('mask', data=mask)


# フォルダー初期化
# for format in formats:
#     if(os.path.exists(format)):
#         shutil.rmtree(format)
