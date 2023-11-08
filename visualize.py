#!/usr/bin/env python
import os
import cv2
import numpy as np


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


import tempfile
from subprocess import call
if 'DISPLAY' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl' #egl
import pyrender
import trimesh
from psbody.mesh import Mesh

from utils import save_video

# The implementation of rendering is borrowed from VOCA: https://github.com/TimoBolkart/voca/blob/master/utils/rendering.py
def render_mesh_helper(mesh, t_center, rot=np.zeros(3), tex_img=None, z_offset=0):
    camera_params = {'c': np.array([400, 400]),
                         'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                         'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center
    
    intensity = 2.0

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.8, 
                roughnessFactor=0.8 
            )


    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)
  
    scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])

    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3,3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())
    
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def main():

    save_folder = "Visualize"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    from dataset.data_loader import get_dataloaders
    class Configuration:
        def __init__(self):
            self.dataset = "vox"
            self.read_audio = True
            self.batch_size = 1

    config = Configuration()
    
    dataset_train = get_dataloaders(config)['train']
    wav, vert, template = next(iter(dataset_train))
    visualize( wav, vert, template, save_folder)


def visualize(wav, vert, template, save_folder, gt=False, fps=30):
    # generate the facial animation (.npy file) for the audio 
    print('Generating facial animation for {}...'.format(1))
    
    prediction = vert.squeeze() # (seq_len, V*3)

    # np.save("styles/3.npy",prediction.detach().cpu().numpy())
    ######################################################################################
    ##### render the npy file

    template_file = 'dataset/model/FLAME_sample.ply'
         
    print("rendering: ", 1)
    # template_file = "demo/Base3.ply"
    template = Mesh(filename=template_file)
    predicted_vertices = prediction.detach().cpu().numpy()
    predicted_vertices = np.reshape(predicted_vertices,(-1,5023,3))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    num_frames = predicted_vertices.shape[0]
    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=save_folder)
    
    writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (800, 800), True)
    center = np.mean(predicted_vertices[0], axis=0)
    
    for i_frame in range(num_frames):
        render_mesh = Mesh(predicted_vertices[i_frame], template.f)
        pred_img = render_mesh_helper(render_mesh, center)
        pred_img = pred_img.astype(np.uint8)
        writer.write(pred_img)

    writer.release()
    id  = wav.split("_codes")[0]
    file_name = id.split('/')[-1]
    id += ".wav"
    video_fname = os.path.join(save_folder, file_name+'.mp4')
    cmd = ('ffmpeg -y' + ' -i {0} -c:v libx264 -pix_fmt yuv420p -qscale 0 {1}'.format(
       tmp_video_file.name, video_fname)).split()
    call(cmd)
    video_path = video_fname.replace('.mp4', '_audio_gt.mp4') if gt else video_fname.replace('.mp4', '_audio.mp4') 
    print("Adding audio to ", id)
    cmd = ('ffmpeg -y' + ' -i {0} -i {1}  -channel_layout stereo -qscale 0 {2}'.format(
       id, video_fname, video_path)).split()
    call(cmd)

    if os.path.exists(video_fname):
        os.remove(video_fname)
    return video_path

def visualize_frames(wav, frames, save_folder, gt=False, fps=30):

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    output_path = os.path.join(save_folder, "out.mp4")
    if gt:
        output_path = output_path.replace('.mp4', '_gt.mp4')
    
    save_video(output_path, frames.detach().cpu().numpy(), fps=fps)
    id  = wav[0].split("_codes")[0]
    id += ".wav"
    cmd = ('ffmpeg' + ' -y -i {0} -i {1}  -channel_layout stereo -qscale 0 {2}'.format(
        id,output_path, output_path.replace('.mp4', '_audio.mp4'))).split()
    call(cmd)
    if os.path.exists(output_path):
        os.remove(output_path)
    
    return output_path.replace('.mp4', '_audio.mp4')

if __name__ == '__main__':
    main()
