if (! Detector.webgl) Detector.addGetWebGLMessage();

var renderer, scene, camera;
var bsize = 200;
var pointSize = bsize * 0.055;
var rotateY = new THREE.Matrix4().makeRotationY(0.01);
var uniforms = {
    time:    { type: "f", value: 0.0 },
    color:   { type: "c", value: new THREE.Color(0xffffff) },
    texture: { type: "t", value: null }
};
var start = Date.now();
var once = true;

var loader = new THREE.TextureLoader();
loader.load('res/particle.png', function (texture) {
    uniforms.texture.value = texture;
    init();

    render();

    $("body").on('touchstart mousedown', function(event) {
    if (! once) { return; }
    once = false;
    start = Date.now();
    uniforms[ 'time' ].value = 0.00025 * (Date.now() - start);
    animate();
    $(document).ready(function () {
        $("div.hiddenlogo").fadeIn(8000).removeClass("hiddenlogo");
        $("div.hiddencontent").fadeIn(10000).removeClass("hiddencontent");
        });
    });
});

function generateDomeCloud() {
    var geometry = new THREE.BufferGeometry();

    var k = 0;
    var pCount = 5000;

    var positions = new Float32Array(pCount * 3);
    var sizes = new Float32Array(pCount);

    for(var j = 0; j < 1; j++) {
        for(var i = 0; i < pCount; i++) {
            var R = 150 - (16 * (j));

            var PHI = (Math.sqrt(5)+1)/2 - 1;     // golden ratio
            var GA = PHI * Math.PI * 2;           // golden angle

            var lon = GA * (i+1);
            lon /= Math.PI * 2;
            lon -= Math.floor(lon);
            lon *= Math.PI * 2;
            if (lon > Math.PI) {lon -= Math.PI * 2;}
            var lat = Math.asin(-1 + 2 * (i+1) / (pCount + 32.0));

            var x = R * Math.cos(lat) * Math.cos(lon);
            var y = R * Math.cos(lat) * Math.sin(lon);
            var z = R * Math.sin(lat);

            sizes[ k ] = pointSize;
            positions[ k*3 ] = x;
            positions[ k*3+1 ] = y;
            positions[ k*3+2 ] = z;

            k++;
        }
    }

    geometry.addAttribute('size', new THREE.BufferAttribute(sizes, 1));
    geometry.addAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.computeBoundingBox();

    var material = new THREE.ShaderMaterial({
        uniforms: uniforms,
        vertexShader: `
            uniform float time;
            attribute float size;
            varying vec3 fN;
            varying vec3 fV;

            // https://github.com/ashima/webgl-noise
            vec3 mod289(vec3 x)
            {
                return x - floor(x * (1.0 / 289.0)) * 289.0;
            }

            vec4 mod289(vec4 x)
            {
                return x - floor(x * (1.0 / 289.0)) * 289.0;
            }

            vec4 permute(vec4 x)
            {
                return mod289(((x*34.0)+1.0)*x);
            }

            vec4 taylorInvSqrt(vec4 r)
            {
                return 1.79284291400159 - 0.85373472095314 * r;
            }

            vec3 fade(vec3 t) {
                return t*t*t*(t*(t*6.0-15.0)+10.0);
            }

            // Classic Perlin noise, periodic variant
            float pnoise(vec3 P, vec3 rep)
            {
                vec3 Pi0 = mod(floor(P), rep); // Integer part, modulo period
                vec3 Pi1 = mod(Pi0 + vec3(1.0), rep); // Integer part + 1, mod period
                Pi0 = mod289(Pi0);
                Pi1 = mod289(Pi1);
                vec3 Pf0 = fract(P); // Fractional part for interpolation
                vec3 Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
                vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
                vec4 iy = vec4(Pi0.yy, Pi1.yy);
                vec4 iz0 = Pi0.zzzz;
                vec4 iz1 = Pi1.zzzz;

                vec4 ixy = permute(permute(ix) + iy);
                vec4 ixy0 = permute(ixy + iz0);
                vec4 ixy1 = permute(ixy + iz1);

                vec4 gx0 = ixy0 * (1.0 / 7.0);
                vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
                gx0 = fract(gx0);
                vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
                vec4 sz0 = step(gz0, vec4(0.0));
                gx0 -= sz0 * (step(0.0, gx0) - 0.5);
                gy0 -= sz0 * (step(0.0, gy0) - 0.5);

                vec4 gx1 = ixy1 * (1.0 / 7.0);
                vec4 gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
                gx1 = fract(gx1);
                vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
                vec4 sz1 = step(gz1, vec4(0.0));
                gx1 -= sz1 * (step(0.0, gx1) - 0.5);
                gy1 -= sz1 * (step(0.0, gy1) - 0.5);

                vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
                vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
                vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
                vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
                vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
                vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
                vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
                vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

                vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
                g000 *= norm0.x;
                g010 *= norm0.y;
                g100 *= norm0.z;
                g110 *= norm0.w;
                vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
                g001 *= norm1.x;
                g011 *= norm1.y;
                g101 *= norm1.z;
                g111 *= norm1.w;

                float n000 = dot(g000, Pf0);
                float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
                float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
                float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
                float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
                float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
                float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
                float n111 = dot(g111, Pf1);

                vec3 fade_xyz = fade(Pf0);
                vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
                vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
                float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
                return 2.2 * n_xyz;
            }
            //

            float turbulence(vec3 p) {
                float w = 100.0;
                float t = -0.5;
                for (float f = 1.0 ; f <= 10.0 ; f++){
                    float power = pow(2.0, f);
                    t += abs(pnoise(vec3(power * p), vec3(10.0, 10.0, 10.0)) / power);
                }
                return t;
            }

            void main() {
                vec3 N = normalize(position);
                float radius = length(position);

                float noise = 1000.0 *  -0.10 * turbulence(0.5 * N + time);
                float b = 5.0 * pnoise(0.05 * position + vec3(2.0 * time), vec3(100.0));
                float displacement = - noise + b + time * 75.;
                displacement = clamp(displacement, 0., 15.);

                vec3 newPosition = position + N * displacement;

                fN = N;
                fV = -vec3(modelViewMatrix*vec4(newPosition, 1.0));

                vec4 mvPosition = modelViewMatrix * vec4(newPosition, 1.0);
                gl_PointSize = size;
                gl_Position = projectionMatrix * mvPosition;
            }
        `,
        fragmentShader: `
            uniform sampler2D texture;
            varying vec3 fN;
            varying vec3 fV;

            void main() {
                float ratio = dot(normalize(fV),normalize(fN));

                gl_FragColor = mix(vec4(1,1,1,1),vec4(0.5,0.5,0.5,1),ratio) * texture2D(texture, gl_PointCoord);
                if (gl_FragColor.a < 0.85) discard;
            }
        `,
    });
    var pointcloud = new THREE.Points(geometry, material);

    return pointcloud;
}

function init() {
    uniforms[ 'time' ].value = 0.0;

    container = document.getElementById('bilye');

    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(35, 1.0, 1, 10000);
    camera.applyMatrix(new THREE.Matrix4().makeTranslation(0,0,600));

    var pcSphere = generateDomeCloud();
    scene.add(pcSphere);

    var geometry = new THREE.SphereGeometry(155, 32, 32);
    var material = new THREE.MeshBasicMaterial({color: 0x888888});
    var sphere = new THREE.Mesh(geometry, material);
    scene.add(sphere);

    renderer = Detector.webgl ? new THREE.WebGLRenderer({ alpha: true }) : new THREE.CanvasRenderer({ alpha: true });
    renderer.setSize(bsize, bsize);
    renderer.setPixelRatio(window.devicePixelRatio ? window.devicePixelRatio : 1)
    container.appendChild(renderer.domElement);
}

function animate() {
    if (uniforms[ 'time' ].value < 1.0) {
        requestAnimationFrame(animate);
    }

    uniforms[ 'time' ].value = 0.00025 * (Date.now() - start);

    render();
}

function render() {
    camera.updateMatrixWorld();
    renderer.render(scene, camera);
}