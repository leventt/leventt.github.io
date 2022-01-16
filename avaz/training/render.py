import os
import math
import numpy as np
import torch
import moderngl
import moderngl_window as mglw
from pyrr import Matrix44
from PIL import Image

from constants import ROOT
from constants import SHADER
from constants import MESH_INDICES
from constants import NEUTRAL_VERTS
from constants import DELTA_VERTS
from constants import EGL
from constants import SHAPES_LEN

from model import getInference


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class Camera:
    def __init__(self, ratio):
        self.fov = 18.0
        self.aspect = ratio
        self.near = 0.01
        self.far = 100.0
        self.theta = np.pi / 2.0
        self.phi = np.pi / 2.0
        self.radius = 10.0
        self.upsign = 1.0
        self.target = np.array([0.0, 0.0, 0.0], np.float32)
        self.orbit(math.radians(-25), math.radians(0))
        self.pan(0, 0)

        self.navigating = False
        self.init = True

        self.LMB = False
        self.MMB = False
        self.RMB = False
        self.MOD = False
        self.MOD1 = False

    def cameraPosition(self):
        height = math.cos(self.phi) * self.radius
        distance = math.sin(self.phi) * self.radius

        return (
            np.array(
                [
                    distance * math.cos(self.theta),
                    height,
                    distance * math.sin(self.theta),
                ]
            )
            + self.target
        )

    def orbit(self, theta, phi):
        self.phi += phi

        twoPi = np.pi * 2.0
        while self.phi > twoPi:
            self.phi -= twoPi
        while self.phi < -twoPi:
            self.phi += twoPi

        if self.phi < np.pi and self.phi > 0.0:
            self.upsign = 1.0
        elif self.phi < -np.pi and self.phi > -2 * np.pi:
            self.upsign = 1.0
        else:
            self.upsign = -1.0

        self.theta += self.upsign * theta

    def pan(self, dx, dy):
        direction = normalize([self.target - self.cameraPosition()])[0]
        right = np.cross(direction, [0.0, self.upsign, 0.0])
        up = np.cross(right, direction)

        self.target += right * dx
        self.target += up * dy

    def zoom(self, distance):
        if self.radius - distance > 0:
            self.radius -= distance

    def projectionMatrix(self):
        self.navigating = False

        return Matrix44.perspective_projection(
            self.fov, self.aspect, self.near, self.far
        )

    def viewatrix(self):
        self.navigating = False

        direction = normalize([self.target - self.cameraPosition()])[0]
        right = np.cross(direction, [0.0, self.upsign, 0.0])
        up = np.cross(right, direction)
        eye = self.cameraPosition()

        return Matrix44.look_at(eye, self.target, up)

    def mouseDragEvent(self, dx, dy):
        self.navigating = True
        self.init = False

        if self.LMB and not self.MOD and not self.MOD1:
            self.orbit(dx * 0.02, -dy * 0.02)
        elif self.MMB or (self.LMB and self.MOD1):
            self.pan(-dx * self.radius * 0.001, dy * self.radius * 0.001)
        elif self.RMB or (self.LMB and self.MOD):
            if abs(dx) > abs(dy):
                self.zoom(-dx * self.radius * 0.01)
            else:
                self.zoom(dy * self.radius * 0.01)

    def mouseScrollEvent(self, delta):
        self.navigating = True

        self.init = False
        self.zoom(delta * (self.radius / 1000.0))


def getValidationVideoTensor(inference, validationMatcap, width=128, height=128):
    if EGL:
        ctx = moderngl.create_standalone_context(require=330, backend="egl")
    else:
        ctx = moderngl.create_standalone_context(require=330)
    prog = ctx.program(**SHADER)

    aspect = width / height
    camera = Camera(aspect)
    projection = prog["projection"]
    view = prog["view"]
    projection.write((camera.projectionMatrix()).astype(np.float32).tobytes())
    view.write((camera.viewatrix()).astype(np.float32).tobytes())
    vbo = ctx.buffer(NEUTRAL_VERTS.astype(np.float32).tobytes())
    indexBuffer = ctx.buffer(MESH_INDICES.astype(np.uint32).tobytes())
    vao = ctx.vertex_array(
        program=prog, content=[(vbo, "3f", "position")], index_buffer=indexBuffer,
    )
    matcap = ctx.texture(
        validationMatcap.size, len(validationMatcap.mode), validationMatcap.tobytes()
    )

    fbo = ctx.framebuffer(
        ctx.renderbuffer((width, height)), ctx.depth_renderbuffer((width, height))
    )
    copyFBO = ctx.framebuffer(ctx.renderbuffer((width, height)))

    ctx.enable(moderngl.DEPTH_TEST)

    frames = []
    for i in range(inference.shape[0]):
        fbo.use()
        ctx.clear(0.18, 0.18, 0.18)
        buff = NEUTRAL_VERTS + np.sum((inference[i] * DELTA_VERTS.T).T, axis=0)
        vbo.write(buff.astype(np.float32).tobytes())
        matcap.use()
        vao.render(moderngl.TRIANGLES)

        ctx.copy_framebuffer(copyFBO, fbo)
        data = copyFBO.read(components=3, alignment=1)
        img = Image.frombytes("RGB", copyFBO.size, data).transpose(
            Image.FLIP_TOP_BOTTOM
        )
        frames.append(
            torch.from_numpy(np.array(img, dtype=np.uint8))
            .permute(2, 1, 0)
            .unsqueeze(0)
        )
    return torch.stack(frames).permute(1, 0, 2, 4, 3)


class PreviewWindow(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "avaz"
    window_size = (864, 486)
    aspect_ratio = 16 / 9
    resizable = False
    samples = 4

    inference = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prevFrame = -1
        self.frame = 0

        self.prog = self.ctx.program(**SHADER)

        self.camera = Camera(self.aspect_ratio)
        self.projection = self.prog["projection"]
        self.view = self.prog["view"]
        self.vbo = self.ctx.buffer(NEUTRAL_VERTS.astype(np.float32).tobytes())
        indexBuffer = self.ctx.buffer(MESH_INDICES.astype(np.uint32).tobytes())
        self.vao = self.ctx.vertex_array(
            program=self.prog,
            content=[(self.vbo, "3f", "position")],
            index_buffer=indexBuffer,
        )
        matcap = Image.open(os.path.join(ROOT, "data", "matcap.png"))
        self.matcap = self.ctx.texture(matcap.size, len(matcap.mode), matcap.tobytes())

    def key_event(self, key, action, modifiers):
        if key == self.wnd.keys.SPACE and action == self.wnd.keys.ACTION_PRESS:
            self.camera.MOD = True
        elif key == self.wnd.keys.SPACE and action == self.wnd.keys.ACTION_RELEASE:
            self.camera.MOD = False

        if key == self.wnd.keys.X and action == self.wnd.keys.ACTION_PRESS:
            self.camera.MOD1 = True
        elif key == self.wnd.keys.X and action == self.wnd.keys.ACTION_RELEASE:
            self.camera.MOD1 = False

    def mouse_drag_event(self, x, y, dx, dy):
        self.camera.LMB = False
        self.camera.MMB = False
        self.camera.RMB = False

        if self.wnd.mouse_states.left:
            self.camera.LMB = True
        elif self.wnd.mouse_states.middle:
            self.camera.MMB = True
        elif self.wnd.mouse_states.right:
            self.camera.RMB = True

        self.camera.mouseDragEvent(dx, dy)

    def render(self, time, frameTime):
        self.ctx.clear(0.18, 0.18, 0.18)
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.frame = int(time * 30.0)
        self.frame = self.frame % self.inference.shape[0]
        if self.frame != self.prevFrame:
            buff = NEUTRAL_VERTS + np.sum(
                (self.inference[self.frame] * DELTA_VERTS.T).T, axis=0
            )
            self.vbo.write(buff.astype(np.float32).tobytes())
        self.prevFrame = self.frame

        if self.camera.navigating or self.camera.init:
            self.projection.write(
                (self.camera.projectionMatrix()).astype(np.float32).tobytes()
            )
            self.view.write((self.camera.viewatrix()).astype(np.float32).tobytes())

        self.matcap.use()
        self.vao.render(moderngl.TRIANGLES)

    @classmethod
    def run(cls):
        mglw.run_window_config(cls)


if __name__ == "__main__":
    from data import getAudioFeatures

    DEVICE = torch.device("cpu")
    TRACED_SCRIPT_PATH = os.path.join(ROOT, "avaz.pt")
    TRACED_SCRIPT = None
    TRACED_SCRIPT = torch.jit.load(TRACED_SCRIPT_PATH)
    TRACED_SCRIPT.eval()
    audioFeatures = getAudioFeatures(os.path.join(ROOT, "data", "empty.wav")).to(DEVICE)
    # PreviewWindow.inference = getInference(TRACED_SCRIPT, audioFeatures)
    PreviewWindow.inference = (
        torch.zeros(audioFeatures.shape[0], SHAPES_LEN).detach().cpu().numpy()
    )
    print(NEUTRAL_VERTS.shape)
    print(NEUTRAL_VERTS.max())
    print(NEUTRAL_VERTS.min())
    PreviewWindow.run()
