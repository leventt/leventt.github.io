if (!Detector.webgl) Detector.addGetWebGLMessage();

var renderer, scene, camera;
var pcSphere;
var rotateY = new THREE.Matrix4().makeRotationY(0.001);
var pratio = window.devicePixelRatio ? window.devicePixelRatio : 1;
var bsize = 300;
var pointSize = pratio;
var uniforms = {
    time: {type: "f", value: 0.0},
    particleSize: {type: "f", value: pointSize * 3.2},
    seed: {type: "f", value: Math.random()},
    waterLevel: {type: "f", value: 3.4},
    colorA: {type: "c", value: new THREE.Color(0x8cf5d1)},
    colorB: {type: "c", value: new THREE.Color(0x196a89)},
    colorC: {type: "c", value: new THREE.Color(0x1b2f1e)},
    colorD: {type: "c", value: new THREE.Color(0x787456)}
};
var start = Date.now();
var once = true;
var gui;
var Controls = function() {
    this.time = uniforms['time'].value;
    this.particleSize = uniforms['particleSize'].value;
    this.seed = uniforms['seed'].value;
    this.waterLevel = uniforms['waterLevel'].value;
    this.colorA = uniforms['colorA'].value.getHex();
    this.colorB = uniforms['colorB'].value.getHex();
    this.colorC = uniforms['colorC'].value.getHex();
    this.colorD = uniforms['colorD'].value.getHex();
    this.resetTime = function() { start = Date.now(); };
};
var controls = new Controls();

$(document).ready(function () {
    init();
    start = Date.now();

    gui = new dat.GUI({autoPlace: false});
    gui.closed = true;
    // gui.add(controls, 'time').step(.01).listen();
    // gui.add(controls, 'particleSize', 0., 20.).step(.01);
    gui.add(controls, 'seed', 0., 100.).step(.001);
    gui.add(controls, 'waterLevel', -1, 10.).step(.001).listen();
    gui.addColor(controls, 'colorA');
    gui.addColor(controls, 'colorB');
    gui.addColor(controls, 'colorC');
    gui.addColor(controls, 'colorD');
    gui.add(controls, 'resetTime');

    var contentContainer = document.getElementById('content');
    contentContainer.appendChild(gui.domElement);

    render();

    if (! once) { return; }
    once = false;
    start = Date.now();

    controls.time = 0.00025 * (Date.now() - start)
    uniforms['time'].value = controls.time;
    animate();

    $("div.hiddenlogo").fadeIn(8000).removeClass("hiddenlogo");
    $("div.hiddencontent").fadeIn(30000).removeClass("hiddencontent");
});

function generateDomeCloud() {
    var geometry = new THREE.BufferGeometry();

    var k = 0;
    var pCount = 75000;
    var positions = new Float32Array(pCount * 3);

    for(var j = 0; j < 1; j++) {
        for(var i = 1; i <= pCount; i++) {
            var R = 150 + 10*j;

            var PHI = (Math.sqrt(5)+1)/2 - 1;     // golden ratio
            var GA = PHI * Math.PI * 2;           // golden angle

            var lon = GA * i;
            lon /= Math.PI * 2;
            lon -= Math.floor(lon);
            lon *= Math.PI * 2;
            if (lon > Math.PI) {lon -= Math.PI * 2;}
            var lat = Math.asin((2 * i) / pCount);

            var x = R * Math.cos(lat) * Math.cos(lon);
            var y = R * Math.cos(lat) * Math.sin(lon);
            var z = R * Math.sin(lat);

            positions[ k*3 ] = (isNaN(x)) ? 0.: x;
            positions[ k*3+1 ] = (isNaN(y)) ? 0.: y;
            positions[ k*3+2 ] = (isNaN(z)) ? 0.: z;

            k++;
        }
    }

    // hs = new Hexasphere(150, 32, 7.);
    // var positions = new Float32Array(hs.tiles.length * 3);
    // var k = 0;
    // console.log(hs.tiles.length);
    // console.log(hs.tiles);
    // for(var i=0; i < hs.tiles.length; i++) {
    //     positions[ i*3 ] = (isNaN(hs.tiles[i].centerPoint.x)) ? 0.: hs.tiles[i].centerPoint.x;
    //     positions[ i*3+1 ] = (isNaN(hs.tiles[i].centerPoint.y)) ? 0.: hs.tiles[i].centerPoint.y;
    //     positions[ i*3+2 ] = (isNaN(hs.tiles[i].centerPoint.z)) ? 0.: hs.tiles[i].centerPoint.z;
    // }

    new THREE.IcosahedronGeometry([100, 1])
    geometry.addAttribute('position', new THREE.BufferAttribute(positions, 3));

    var material = new THREE.ShaderMaterial({
        uniforms: uniforms,
        vertexShader: `
            uniform float waterLevel;
            uniform float time;
            uniform float particleSize;
            uniform float seed;
            varying vec3 fN;
            varying vec3 fV;
            varying float fDisp;
            varying vec3 fLDir;

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

            // http://www.neilmendoza.com/glsl-rotation-about-an-arbitrary-axis/
            mat4 rotationMatrix(vec3 axis, float angle)
            {
                axis = normalize(axis);
                float s = sin(angle);
                float c = cos(angle);
                float oc = 1.0 - c;
                
                return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                            oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                            oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                            0.0,                                0.0,                                0.0,                                1.0);
            }

            void main() {
                vec3 N = normalize(position);

                float rtime = pow(time + .001, .25);
                float noise = clamp(40., 4., 9.) * turbulence(N + seed) - .7;
                float b = 0.1 * pnoise(0.5 * position + vec3(2.0 * seed), vec3(10000.));
                float displacement = noise + b;
                displacement = pow(abs(noise), 1.1) + b * 3.5;
                displacement = clamp(displacement, 0., 10.);

                if (displacement <= 1.) {
                    float nW = turbulence(N + rtime);
                    displacement += nW * .1;
                }

                vec3 newPosition = position + N * displacement;
                fDisp = displacement;

                vec4 mvPosition = modelViewMatrix * vec4(newPosition, 1.0);
                fV = mvPosition.xyz;
                fN = N;

                fLDir = normalize(vec3(300, 100, -300));
                fLDir = (rotationMatrix(normalize(vec3(-.3, 1., -.1)), rtime * 1.3) * vec4(fLDir, 1.)).xyz;
                fLDir = normalize(fLDir);

                gl_PointSize = particleSize;
                gl_Position = projectionMatrix * mvPosition;
            }
        `,
        fragmentShader: `
            uniform vec3 colorA;
            uniform vec3 colorB;
            uniform vec3 colorC;
            uniform vec3 colorD;
            uniform float waterLevel;
            varying vec3 fN;
            varying vec3 fV;
            varying float fDisp;
            varying vec3 fLDir;

            void main() {
                float intensity = max(dot(fN, fLDir), 0.0);
                float specr = 0.;
                float spec = 0.;
                if (intensity > 0.) {
                    vec3 h = normalize(fLDir + fV);  
                    specr = max(dot(h, fN), 0.0);
                    spec = .05 * pow(abs(specr), 0.1);
                }

                float ratio = dot(normalize(fV), normalize(fN));
                float dratio = pow(abs(fDisp / 5.5), 2.5);
                vec3 diffuse = mix(vec3(.7, .8, .9), mix(colorC, colorD, dratio), clamp(pow(abs(ratio), .25), 0.1, 1.));
                if (fDisp <= waterLevel) {
                    diffuse = mix(vec3(1.), mix(colorB, colorA, dratio), clamp(pow(abs(ratio), .85), 0.1, 1.));
                    if (intensity > 0.) {
                        spec = .35 * pow(abs(specr), 1.5);
                    }
                }

                gl_FragColor = vec4(max(intensity * diffuse + spec, vec3(.01, .02, .03)), 1.);
            }
        `,
    });

    return new THREE.Points(geometry, material);
}

function init() {
    controls.time = 0.;
    uniforms['time'].value = controls.time;

    container = document.getElementById('seyyare');

    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(35, 1.0, 1, 10000);
    camera.applyMatrix(new THREE.Matrix4().makeTranslation(0,0,600));

    pcSphere = generateDomeCloud();
    scene.add(pcSphere);

    var geometry = new THREE.SphereGeometry(170, 32, 32);
    var material = new THREE.MeshBasicMaterial({color: 0xbfffe9, transparent: true, opacity: .05});
    var sphere = new THREE.Mesh(geometry, material);
    sphere.scale.set(1, 1, .1);
    scene.add(sphere);

    var geometry = new THREE.SphereGeometry(175, 32, 32);
    var material = new THREE.MeshBasicMaterial({color: 0xffffff, transparent: true, opacity: .02});
    var sphere = new THREE.Mesh(geometry, material);
    sphere.scale.set(1, 1, .1);
    scene.add(sphere);

    renderer = Detector.webgl ? new THREE.WebGLRenderer({ alpha: true, antialiasing: true }) : new THREE.CanvasRenderer({ alpha: true, antialiasing: true });
    renderer.setSize(bsize, bsize);
    renderer.setPixelRatio(pratio);
    container.appendChild(renderer.domElement);
}

function animate() {
    requestAnimationFrame(animate);

    // pcSphere.applyMatrix(rotateY);

    controls.time = 0.00005 * (Date.now() - start);
    uniforms['time'].value = controls.time;
    uniforms['particleSize'].value = controls.particleSize;
    uniforms['seed'].value = controls.seed;
    uniforms['waterLevel'].value = controls.waterLevel;
    uniforms['colorA'].value.set(controls.colorA);
    uniforms['colorB'].value.set(controls.colorB);
    uniforms['colorC'].value.set(controls.colorC);
    uniforms['colorD'].value.set(controls.colorD);

    render();
}

function render() {
    camera.updateMatrixWorld();
    renderer.render(scene, camera);
}
