import os
import torch
import numpy as np

# TODO: try without this when deploying
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
# OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the
# program. That is dangerous, since it can degrade performance or cause incorrect results.
# The best thing to do is to ensure that only a single OpenMP runtime is linked into the
# process, e.g. by avoiding static linking of the OpenMP runtime in any library.
# As an unsafe, unsupported, undocumented workaround you can set the environment variable
# KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute,
# but that may cause crashes or silently produce incorrect results.
# For more information, please see http://www.intel.com/software/products/support/.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

ROOT = os.getenv("ROOT") or os.path.dirname(__file__)
EGL = os.getenv("EGL") or False
if isinstance(EGL, str):
    EGL = eval(EGL)

DEVICE = torch.device("cuda")
FPS = 30
SAMPLE_RATE = 16000
AUDIOSEQ_LEN = 64
WINDOW_SIZE = 1000 // 2
WINDOW_STRIDE = 1000 / FPS
EXTRACTOR_LEN = 32
FORMANT_LEN = EXTRACTOR_LEN

# SHAPES
# 1 Drop_C
# 2 Open_C
# 3 Roll_C
# 4 Suck_C
# 5 Shrug_C
# 6 FunnelO_C
# 7 FunnelU_C
# 8 Puck_C
# 9 Wide_C
# 10 Frown_C
# 11 Frown_L
# 12 Frown_R
# 13 Dimple_C
# 14 Dimple_L
# 15 Dimple_R
# 16 Down_C
# 17 Up_C
# 18 Left_C
# 19 Right_C
SHAPES_LEN = 9  # 10-19 is not used for training or inference

SHADER = {}
SHADER.update(
    vertex_shader="""
            #version 330 core
            in vec3 position;
            uniform mat4 view;
            uniform mat4 projection;
            out vec3 vPosition;
            void main()
            {
                vPosition = position;
                gl_Position = projection * view * vec4(vPosition, 1.);
            }
            """,
    geometry_shader="""
            #version 330 core
            layout(triangles) in;
            layout(triangle_strip, max_vertices = 3) out;
            in vec3 vPosition[];
            uniform mat4 view;
            out vec3 gNormal;
            out vec3 gBarycentric;
            out vec3 gPosition;
            void main()
            {
                vec3 flatNormal = cross(
                    vPosition[1] - vPosition[0],
                    vPosition[2] - vPosition[0]
                );
                gNormal = normalize(transpose(inverse(mat3(view))) * flatNormal);
                gPosition = vPosition[0];
                gBarycentric = vec3(1, 0, 0);
                gl_Position = gl_in[0].gl_Position; EmitVertex();
                gPosition = vPosition[1];
                gBarycentric = vec3(0, 1, 0);
                gl_Position = gl_in[1].gl_Position; EmitVertex();
                gPosition = vPosition[2];
                gBarycentric = vec3(0, 0, 1);
                gl_Position = gl_in[2].gl_Position; EmitVertex();
                EndPrimitive();
            }
            """,
    fragment_shader="""
            #version 330 core
            in vec3 gPosition;
            in vec3 gNormal;
            in vec3 gBarycentric;
            uniform mat4 view;
            uniform sampler2D matcap;
            out vec4 fragColor;
            void main()
            {
                /*
                if(gBarycentric.x < 0.001 || gBarycentric.y < 0.001 || gBarycentric.z < 0.001) {
                    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
                    vec3 r = reflect(normalize(view * vec4(gPosition, 1.)).xyz, gNormal);
                    r.y *= -1;
                    float m = 2. * sqrt(pow(r.x, 2.) + pow(r.y, 2.) + pow(r.z + 1., 2.));
                    vec2 matcapUV = r.xy / m + .5;
                    vec3 color = texture(matcap, matcapUV.xy).rgb;
                    fragColor = vec4(vec3(color.r) * 2.0, 1.);
                } else { */
                    vec3 r = reflect(normalize(view * vec4(gPosition, 1.)).xyz, gNormal);
                    r.y *= -1;
                    float m = 2. * sqrt(pow(r.x, 2.) + pow(r.y, 2.) + pow(r.z + 1., 2.));
                    vec2 matcapUV = r.xy / m + .5;
                    vec3 color = texture(matcap, matcapUV.xy).rgb;
                    fragColor = vec4(vec3(color.r), 1.);
                //}
            }
            """,
)

MESH_INDICES = np.load(os.path.join(ROOT, "data", "indices.npy"))
NEUTRAL_VERTS = np.load(os.path.join(ROOT, "data", "neutral.npy"))
# (19, 886, 3) we only need first 9 shapes
DELTA_VERTS = np.load(os.path.join(ROOT, "data", "deltas.npy"))[:9]
# z up to y up
t = NEUTRAL_VERTS[..., 1].copy()
NEUTRAL_VERTS[..., 1] = NEUTRAL_VERTS[..., 2]
NEUTRAL_VERTS[..., 2] = -1 * t
t = DELTA_VERTS[..., 1].copy()
DELTA_VERTS[..., 1] = DELTA_VERTS[..., 2]
DELTA_VERTS[..., 2] = -1 * t
