"""
Renders mesh using OpenDr for visualization.
"""
import sys
import numpy as np
import cv2
import pdb
from PIL import Image, ImageDraw
from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from PIL import Image, ImageDraw
import subprocess as sp


colors = {
    # the format is BGR
    'light_blue': [1.0, 128/255, 0],
    'light_pink': [.9, .7, .7],  # This is used to do no-3d
    'light_green': [166/255.0, 178/255.0, 30/255.0],
    'light_purple_han': [0.8, 0.53, 0.53],
    'light_purple_rongyu': [255.0/255.0, 102/255, 102/255],
    'light_gray': [192/255, 192/255, 192/255],
}


def render_together(verts_list, faces_list, color_list, cam, inputSize, img=None):
    assert len(verts_list) == 2, "Current version only support 2 sets of mesh"
    assert len(verts_list) == len(faces_list)
    assert len(faces_list) == len(color_list)

    verts0, verts1 = verts_list
    faces0, faces1 = faces_list
    color0, color1 = color_list
    assert color0.shape == (1, 3) and color1.shape == (1,3)

    verts = np.concatenate((verts0, verts1), axis=0)
    faces = np.concatenate((faces0, faces1+verts0.shape[0]), axis=0)
    color0 = np.repeat(color0, verts0.shape[0], axis=0)
    color1 = np.repeat(color1, verts1.shape[0], axis=0)
    color = np.concatenate((color0, color1), axis=0)

    return render(verts, faces, cam, inputSize, img=img, color=color)


def render(verts, faces, cam, inputSize, img=None, color=None, focal_length=5):
    # assert colors.shape == verts.shape

    # f = 100
    f = focal_length
    tz = f / cam[0]
    cam_for_render = 0.5 * inputSize * np.array([f, 1, 1])
    cam_t = np.array([cam[1], cam[2], tz])

    verts = verts+cam_t
    cam = cam_for_render

    if img is None:
        img = np.ones((inputSize, inputSize, 3), dtype=np.float32)
    
    renderer = SMPLRenderer(img_size=inputSize)
    rend_img = renderer(
        img=img, cam=cam, 
        verts=verts, faces=faces,
        color = color)
    return rend_img


class SMPLRenderer(object):
    def __init__(self,
                 img_size=224,
                 flength=500.
    ):
        self.w = img_size
        self.h = img_size
        self.flength = flength

    def __call__(self,
                 verts,
                 faces,
                 color=None,
                 cam=None,
                 img=None,
                 do_alpha=False,
                 far=None,
                 near=None,
                 img_size=None):
        """
        cam is 3D [f, px, py]
        """
        if img is not None:
            h, w = img.shape[:2]
        elif img_size is not None:
            h = img_size[0]
            w = img_size[1]
        else:
            h = self.h
            w = self.w

        if cam is None:
            cam = [self.flength, w / 2., h / 2.]

        use_cam = ProjectPoints(
            f=cam[0] * np.ones(2),
            rt=np.zeros(3),
            t=np.zeros(3),
            k=np.zeros(5),
            c=cam[1:3])

        if near is None:
            near = np.maximum(np.min(verts[:, 2]) - 1, 0.1)

        if far is None:
            far = np.maximum(np.max(verts[:, 2]) + 1, 25)
            far = np.max(verts[:, 2]) + 1

        imtmp = render_model(
            verts,
            faces,
            w,
            h,
            use_cam,
            do_alpha=do_alpha,
            img=img,
            far=far,
            near=near,
            color = color)
        image = (imtmp * 255).astype('uint8')

        return image


def _create_renderer(w=640,
                     h=480,
                     rt=np.zeros(3),
                     t=np.zeros(3),
                     f=None,
                     c=None,
                     k=None,
                     near=.5,
                     far=10.):

    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5) if k is None else k

    rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}
    return rn



def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                   [-np.sin(angle), 0., np.cos(angle)]])
    return np.dot(points, ry)


def simple_renderer(rn,
                    verts,
                    faces,
                    yrot=np.radians(120),
                    color=colors['light_pink']):
    # Rendered model color
    rn.set(v=verts, f=faces, vc=color, bgcolor=np.ones(3))
    albedo = rn.vc

    # Construct Back Light (on back right corner)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Right Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=np.array([.7, .7, .7]))
    
    return rn.r


def get_alpha(imtmp, bgval=1.):
    h, w = imtmp.shape[:2]
    alpha = (~np.all(imtmp == bgval, axis=2)).astype(imtmp.dtype)

    b_channel, g_channel, r_channel = cv2.split(imtmp)

    im_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha.astype(
        imtmp.dtype)))
    return im_RGBA


def append_alpha(imtmp):
    alpha = np.ones_like(imtmp[:, :, 0]).astype(imtmp.dtype)
    if np.issubdtype(imtmp.dtype, np.uint8):
        alpha = alpha * 255
    b_channel, g_channel, r_channel = cv2.split(imtmp)
    im_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha))
    return im_RGBA



def render_model(verts,
                 faces,
                 w,
                 h,
                 cam,
                 near=0.5,
                 far=25,
                 img=None,
                 do_alpha=False,
                 color=None):
    rn = _create_renderer(
        w=w, h=h, near=near, far=far, rt=cam.rt, t=cam.t, f=cam.f, c=cam.c)

    # Uses img as background, otherwise white background.
    if img is not None:
        rn.background_image = img / 255. if img.max() > 1 else img

    if color is None:
        color = colors['light_blue']

    imtmp = simple_renderer(rn, verts, faces, color=color)

    # If white bg, make transparent.
    if img is None and do_alpha:
        imtmp = get_alpha(imtmp)
    elif img is not None and do_alpha:
        imtmp = append_alpha(imtmp)

    return imtmp