import blenderproc as bproc
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', nargs='?', help="Path to the bop datasets parent directory")
parser.add_argument('cc_textures_path', nargs='?', default="resources/cctextures", help="Path to downloaded cc textures")
parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved")
args = parser.parse_args()

DATASET_NAME = "z_bracket"

bproc.init()

# z_bracket 객체만 로드 (distractor 없음)
sampled_bop_objs = bproc.loader.load_bop_objs(
    bop_dataset_path=os.path.join(args.bop_parent_path, DATASET_NAME),
    mm2m=True,
    sample_objects=True,
    num_of_objs_to_sample=20)

# 카메라 intrinsics 로드 (dataset의 camera.json 사용)
bproc.loader.load_bop_intrinsics(bop_dataset_path=os.path.join(args.bop_parent_path, DATASET_NAME))

# 재질·물리 설정 (반사 금속)
for obj in sampled_bop_objs:
    obj.enable_rigidbody(True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
    obj.set_shading_mode('auto')
    mat = obj.get_materials()[0]
    # 반사 금속: metallic 높게, roughness 낮게~중간
    mat.set_principled_shader_value("Metallic", np.random.uniform(0.7, 1.0))
    mat.set_principled_shader_value("Roughness", np.random.uniform(0.1, 0.5))
    grey_col = np.random.uniform(0.3, 0.9)
    mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])

# 빈(bin) 생성 — 44 x 34 x 10 cm
bin_x = 0.22   # 가로 반너비 (44cm)
bin_y = 0.17   # 세로 반너비 (34cm)
bin_h = 0.10   # 벽 높이 (10cm)

room_planes = [
    # 바닥
    bproc.object.create_primitive('PLANE', scale=[bin_x, bin_y, 1]),
    # 앞뒤 벽 (y 방향)
    bproc.object.create_primitive('PLANE', scale=[bin_x, bin_h, 1],
        location=[0, -bin_y, bin_h], rotation=[-1.570796, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[bin_x, bin_h, 1],
        location=[0, bin_y, bin_h], rotation=[1.570796, 0, 0]),
    # 좌우 벽 (x 방향)
    bproc.object.create_primitive('PLANE', scale=[bin_h, bin_y, 1],
        location=[bin_x, 0, bin_h], rotation=[0, 1.570796, 0]),
    bproc.object.create_primitive('PLANE', scale=[bin_h, bin_y, 1],
        location=[-bin_x, 0, bin_h], rotation=[0, -1.570796, 0]),
]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', friction=100.0, linear_damping=0.99, angular_damping=0.99)

# 조명
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
light_plane_material.make_emissive(emission_strength=np.random.uniform(3, 6),
                                   emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))
light_plane.replace_materials(light_plane_material)

light_point = bproc.types.Light()
light_point.set_energy(200)
light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
location = bproc.sampler.shell(center=[0, 0, 0], radius_min=1, radius_max=1.5,
                               elevation_min=5, elevation_max=89, uniform_volume=False)
light_point.set_location(location)

# CC 텍스처
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)
random_cc_texture = np.random.choice(cc_textures)
for plane in room_planes:
    plane.replace_materials(random_cc_texture)

# 물체 초기 위치 샘플링 (박스 안 좁은 범위)
def sample_pose_func(obj: bproc.types.MeshObject):
    obj.set_location(np.random.uniform(
        [-0.18, -0.13, 0.1],   # 벽에서 여유 두고
        [0.18, 0.13, 0.3]      # 빈 위에서 떨어짐
    ))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

bproc.object.sample_poses(objects_to_sample=sampled_bop_objs,
                          sample_pose_func=sample_pose_func,
                          max_tries=1000)

# 물리 시뮬레이션 (떨어뜨려 쌓기)
bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                  max_simulation_time=10,
                                                  check_object_interval=1,
                                                  substeps_per_frame=20,
                                                  solver_iters=25)

bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_bop_objs)

# 카메라: 탑다운, 높이 ~0.94m, 약간의 변화
poses = 0
while poses < 25:
    location = np.array([
        np.random.uniform(-0.08, 0.08),
        np.random.uniform(-0.08, 0.08),
        np.random.uniform(0.4, 0.5)
    ])
    # 관심점: 물체들 중심
    poi = bproc.object.compute_poi(np.random.choice(sampled_bop_objs, size=min(10, len(sampled_bop_objs))))
    # 아래를 바라보게 (약간의 inplane 회전)
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.3, 0.3))
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)
    poses += 1

# 렌더링
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)
data = bproc.renderer.render()

# BOP 포맷 저장
bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
                       dataset=DATASET_NAME,
                       depths=data["depth"],
                       colors=data["colors"],
                       color_file_format="JPEG",
                       ignore_dist_thres=10)