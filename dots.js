const IMAGE_PROCESS_SHADER = /*wgsl*/`
@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> params: vec4<f32>; // blank_level, bmp_charge, charge_scale, 0

@compute @workgroup_size(8, 8, 1)
fn process_image(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }
    
    let uv = vec2<f32>(f32(global_id.x) + 0.5, f32(global_id.y) + 0.5) / vec2<f32>(dims);
    let color = textureSampleLevel(input_texture, input_sampler, uv, 0.0);

    // luminance
    let gray = 0.2989 * color.r + 0.5870 * color.g + 0.1140 * color.b;
    
    // (blankLevel - gray) * bmpCharge * chargeScale
    let field_charge = (params.x - gray) * params.y * params.z;

    textureStore(output_texture, global_id.xy, vec4<f32>(field_charge, field_charge, field_charge, 1.0));
}
`;

const COMPUTE_SHADER = /*wgsl*/`
struct SimParams {
    num_dots: u32,
    dot_charge: f32,
    blank_level: f32,
    time_step: f32,
    max_velocity: f32,
    max_displacement: f32,
    sustain: f32,
    image_width: u32,
    image_height: u32,
    time: f32,
    bmp_charge: f32,
    aspect_ratio: f32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> accelerations: array<vec2<f32>>;
@group(0) @binding(4) var input_texture: texture_2d<f32>;
@group(0) @binding(5) var input_sampler: sampler;

@compute @workgroup_size(256, 1, 1)
fn update_acceleration(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_dots) {
        return;
    }
    
    let pos = positions[idx];
    var acc = vec2<f32>(0.0, 0.0);
    let epsilon = 0.00003;
    
    // force from the underlying field
    let sim_xy = min(192, i32(sqrt(f32(params.num_dots)) * 2.0));
    let tex_dims = vec2<f32>(f32(params.image_width), f32(params.image_height));
    
    let stride_x = tex_dims.x / f32(sim_xy);
    let stride_y = tex_dims.y / f32(sim_xy);
    let scale = (stride_x / tex_dims.x) * (stride_y / tex_dims.y) * tex_dims.x * tex_dims.y;
    
    for (var y = 0; y < sim_xy; y++) {
        for (var x = 0; x < sim_xy; x++) {
            let fx = f32(x) * stride_x;
            let fy = f32(y) * stride_y;
            
            let jitter_x = sin(f32(idx) * 12.9898 + f32(x) * 78.233) * 0.5;
            let jitter_y = cos(f32(idx) * 12.9898 + f32(y) * 78.233) * 0.5;
            
            let tx = fx + jitter_x;
            let ty = fy + jitter_y;
            
            var bmp_xy: vec2<f32>;
            if (params.aspect_ratio >= 1.0) {
                bmp_xy.x = (tx - tex_dims.x * 0.5) / (tex_dims.x * 0.5) * params.aspect_ratio;
                bmp_xy.y = (ty - tex_dims.y * 0.5) / (tex_dims.y * 0.5);
            } else {
                bmp_xy.x = (tx - tex_dims.x * 0.5) / (tex_dims.x * 0.5);
                bmp_xy.y = (ty - tex_dims.y * 0.5) / (tex_dims.y * 0.5) / params.aspect_ratio;
            }
            
            let uv = vec2<f32>(tx / tex_dims.x, 1.0 - ty / tex_dims.y);
            let bmp_q = textureSampleLevel(input_texture, input_sampler, uv, 0.0).r * scale;
            
            var dp = bmp_xy - pos;
            let bounds_x = max(1.0, params.aspect_ratio);
            let bounds_y = max(1.0, 1.0 / params.aspect_ratio);
            if (dp.x > bounds_x) { dp.x -= 2.0 * bounds_x; }
            else if (dp.x < -bounds_x) { dp.x += 2.0 * bounds_x; }
            if (dp.y > bounds_y) { dp.y -= 2.0 * bounds_y; }
            else if (dp.y < -bounds_y) { dp.y += 2.0 * bounds_y; }
            
            let d2 = dot(dp, dp) + 0.00003;
            let d = sqrt(d2);
            
            if (d > 0.0000001) {
                let q = bmp_q / d2;
                acc += q * dp / d;
            }
        }
    }
    
    // repulsion
    let cutoff_dist = 0.2;
    let cutoff_dist2 = cutoff_dist * cutoff_dist;
    for (var i = 0u; i < params.num_dots; i++) {
        if (i != idx) {
            let other_pos = positions[i];
            var dp = other_pos - pos;
            
            if (abs(dp.x) > cutoff_dist || abs(dp.y) > cutoff_dist) { continue; }
            
            let bounds_x = max(1.0, params.aspect_ratio);
            let bounds_y = max(1.0, 1.0 / params.aspect_ratio);
            if (dp.x > bounds_x) { dp.x -= 2.0 * bounds_x; }
            else if (dp.x < -bounds_x) { dp.x += 2.0 * bounds_x; }
            if (dp.y > bounds_y) { dp.y -= 2.0 * bounds_y; }
            else if (dp.y < -bounds_y) { dp.y += 2.0 * bounds_y; }
            
            let d2 = dot(dp, dp) + epsilon;
            
            if (d2 > cutoff_dist2) { continue; }
            
            let d = sqrt(d2);
            let q = params.dot_charge / d2;
            acc -= q * dp / d;
        }
    }
    
    accelerations[idx] = acc * params.dot_charge;
}

@compute @workgroup_size(256, 1, 1)
fn update_velocity(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_dots) {
        return;
    }
    
    let acc = accelerations[idx];
    var vel = velocities[idx];
    
    let dt_2 = params.time_step * 0.5;
    vel = vel * params.sustain + acc * dt_2 * 2.0;
    
    let speed = length(vel);
    if (speed > params.max_velocity) {
        vel = vel * (params.max_velocity / speed);
    }
    
    velocities[idx] = vel;
}

@compute @workgroup_size(256, 1, 1)
fn update_position(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_dots) {
        return;
    }
    
    var pos = positions[idx];
    let vel = velocities[idx];
    let acc = accelerations[idx];
    
    let dt = params.time_step;
    let dtt = dt * dt;
    var delta = vel * dt + 0.5 * acc * dtt;
    
    let delta_length = length(delta);
    if (delta_length > params.max_displacement) {
        delta = delta * (params.max_displacement / delta_length);
    }
    
    pos += delta;
    
    let bounds_x = max(1.0, params.aspect_ratio);
    let bounds_y = max(1.0, 1.0 / params.aspect_ratio);
    if (pos.x < -bounds_x) { pos.x += 2.0 * bounds_x; }
    else if (pos.x > bounds_x) { pos.x -= 2.0 * bounds_x; }
    if (pos.y < -bounds_y) { pos.y += 2.0 * bounds_y; }
    else if (pos.y > bounds_y) { pos.y -= 2.0 * bounds_y; }
    
    positions[idx] = pos;
}
`;

const RENDER_SHADER = /*wgsl*/`
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) field_charge: f32,
};

struct RenderParams {
    resolution: vec2<f32>,
    dot_size: f32,
    aspect_ratio: f32,
    size_modulation: f32,
    padding: vec3<f32>,
};

@group(0) @binding(0) var<uniform> render_params: RenderParams;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var input_texture: texture_2d<f32>;
@group(0) @binding(3) var texture_sampler: sampler;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let vertices = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );
    
    let vertex = vertices[vertex_idx];
    let pos = positions[instance_idx];
    
    var sample_uv: vec2<f32>;
    if (render_params.aspect_ratio >= 1.0) {
        sample_uv.x = (pos.x / render_params.aspect_ratio + 1.0) * 0.5;
        sample_uv.y = 1.0 - (pos.y + 1.0) * 0.5;
    } else {
        sample_uv.x = (pos.x + 1.0) * 0.5;
        sample_uv.y = 1.0 - (pos.y * render_params.aspect_ratio + 1.0) * 0.5;
    }
    
    let color = textureSampleLevel(input_texture, texture_sampler, sample_uv, 0.0);
    let gray = 0.2989 * color.r + 0.5870 * color.g + 0.1140 * color.b;
    let normalized_darkness = 1.0 - gray;
    
    let modulated_size = 0.2 + normalized_darkness * 2.3;
    let size_factor = mix(1.0, modulated_size, render_params.size_modulation);
    
    let base_radius = render_params.dot_size * 2.0 / min(render_params.resolution.x, render_params.resolution.y);
    let dot_radius = base_radius * size_factor;
    
    var output: VertexOutput;
    var display_pos = pos;
    let viewport_aspect = render_params.resolution.x / render_params.resolution.y;
    
    var dot_offset = vertex * dot_radius;
    if (viewport_aspect > 1.0) {
        dot_offset.x = dot_offset.x / viewport_aspect;
    } else {
        dot_offset.y = dot_offset.y * viewport_aspect;
    }
    
    if (render_params.aspect_ratio >= 1.0) {
        display_pos.x = display_pos.x / render_params.aspect_ratio;
        if (viewport_aspect < render_params.aspect_ratio) {
            let scale = viewport_aspect / render_params.aspect_ratio;
            display_pos.y = display_pos.y * scale;
        }
    } else {
        display_pos.y = display_pos.y * render_params.aspect_ratio;
        if (viewport_aspect > 1.0 / render_params.aspect_ratio) {
            let scale = (1.0 / render_params.aspect_ratio) / viewport_aspect;
            display_pos.x = display_pos.x * scale;
        }
    }
    
    output.position = vec4<f32>(display_pos.x + dot_offset.x, display_pos.y + dot_offset.y, 0.0, 1.0);
    output.uv = vertex;
    output.field_charge = normalized_darkness;
    return output;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist2 = dot(in.uv, in.uv);
    if (dist2 > 1.0) {
        discard;
    }
    
    return vec4<f32>(0.23, 0.23, 0.23, 1.0);
}
`;

class BlueNoiseStippling {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('webgpu');

        this.numDots = 8192;
        this.dotCharge = 0.2;
        this.blankLevel = 0.95;
        this.timeStep = 0.001;
        this.maxDisplacement = 0.005;
        this.maxVelocity = this.maxDisplacement / this.timeStep;
        this.sustain = 0.95;
        this.dotSize = 1.0;
        this.sizeModulation = 0.34;
        this.bmpCharge = 1.0;
        this.chargeScale = 0.13;

        this.imageWidth = 512;
        this.imageHeight = 512;
        this.aspectRatio = 1.0;

        this.time = 0.0;
        this.running = true;

        this.init();
    }

    async init() {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }

        this.adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance'
        });

        if (!this.adapter) {
            throw new Error('No WebGPU adapter found');
        }

        this.device = await this.adapter.requestDevice();

        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.ctx.configure({
            device: this.device,
            format: this.format,
            alphaMode: 'opaque'
        });

        this.createTextures();
        this.initParticles();
        await this.createDefaultImage();
        this.uploadImage();
        this.processImageGPU();
        this.createPipelines();

        this.animate();
    }

    async createDefaultImage() {
        const canvas = document.createElement('canvas');
        canvas.width = 432;
        canvas.height = 243;
        const ctx = canvas.getContext('2d');

        const gradient = ctx.createRadialGradient(432, 243, 0, 432, 243, 300);
        gradient.addColorStop(0, 'white');
        gradient.addColorStop(1, 'black');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 432, 243);

        const imageData = ctx.getImageData(0, 0, 432, 243);
        await this.setImageData(imageData);
    }

    async setImageData(imageData) {
        this.imageWidth = imageData.width;
        this.imageHeight = imageData.height;
        this.aspectRatio = this.imageWidth / this.imageHeight;

        this.canvas.width = this.imageWidth;
        this.canvas.height = this.imageHeight;

        const rgbaData = new Uint8Array(this.imageWidth * this.imageHeight * 4);
        for (let i = 0; i < imageData.data.length; i++) {
            rgbaData[i] = imageData.data[i];
        }

        let bmpChargeTotal = 0;
        for (let i = 0; i < rgbaData.length; i += 4) {
            const gray = (0.2989 * rgbaData[i] + 0.5870 * rgbaData[i + 1] + 0.1140 * rgbaData[i + 2]) / 255.0;
            bmpChargeTotal += this.blankLevel - gray;
        }

        const dotChargeTotal = this.numDots * this.dotCharge;
        this.bmpCharge = bmpChargeTotal > 0 ? dotChargeTotal / bmpChargeTotal : 1.0;

        this.imageData = rgbaData;

        if (this.device && this.inputTexture) {
            const textureInfo = this.inputTexture;
            if (textureInfo.width !== this.imageWidth || textureInfo.height !== this.imageHeight) {
                this.createTextures();
                this.updateComputeBindGroup();
            }
            this.uploadImage();
            this.processImageGPU();
        }
    }

    initParticles() {
        const positions = new Float32Array(this.numDots * 2);
        for (let i = 0; i < this.numDots; i++) {
            if (this.aspectRatio >= 1.0) {
                positions[i * 2] = (Math.random() * 2 - 1) * this.aspectRatio;
                positions[i * 2 + 1] = Math.random() * 2 - 1;
            } else {
                positions[i * 2] = Math.random() * 2 - 1;
                positions[i * 2 + 1] = (Math.random() * 2 - 1) / this.aspectRatio;
            }
        }

        const velocities = new Float32Array(this.numDots * 2);
        const accelerations = new Float32Array(this.numDots * 2);

        if (this.device) {
            this.positionBuffer = this.device.createBuffer({
                size: positions.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true
            });
            new Float32Array(this.positionBuffer.getMappedRange()).set(positions);
            this.positionBuffer.unmap();

            this.velocityBuffer = this.device.createBuffer({
                size: velocities.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true
            });
            new Float32Array(this.velocityBuffer.getMappedRange()).set(velocities);
            this.velocityBuffer.unmap();

            this.accelerationBuffer = this.device.createBuffer({
                size: accelerations.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true
            });
            new Float32Array(this.accelerationBuffer.getMappedRange()).set(accelerations);
            this.accelerationBuffer.unmap();

            this.updateSimParams();
        }
    }

    createTextures() {
        this.inputTexture = this.device.createTexture({
            size: [this.imageWidth, this.imageHeight],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        });

        this.fieldTexture = this.device.createTexture({
            size: [this.imageWidth, this.imageHeight],
            format: 'rgba32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
        });

        this.sampler = this.device.createSampler({
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
            magFilter: 'nearest',
            minFilter: 'nearest'
        });
    }

    uploadImage() {
        if (this.imageData) {
            this.device.queue.writeTexture(
                { texture: this.inputTexture },
                this.imageData,
                { bytesPerRow: 4 * this.imageWidth },
                [this.imageWidth, this.imageHeight]
            );
        }
    }

    processImageGPU() {
        if (!this.imageProcessPipeline) {
            const shader = this.device.createShaderModule({
                code: IMAGE_PROCESS_SHADER
            });

            this.imageProcessPipeline = this.device.createComputePipeline({
                layout: 'auto',
                compute: {
                    module: shader,
                    entryPoint: 'process_image'
                }
            });
        }

        const params = new Float32Array([
            this.blankLevel,
            this.bmpCharge,
            this.chargeScale,
            0.0
        ]);

        const paramsBuffer = this.device.createBuffer({
            size: params.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(paramsBuffer.getMappedRange()).set(params);
        paramsBuffer.unmap();

        const bindGroup = this.device.createBindGroup({
            layout: this.imageProcessPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.inputTexture.createView() },
                { binding: 1, resource: this.sampler },
                { binding: 2, resource: this.fieldTexture.createView() },
                { binding: 3, resource: { buffer: paramsBuffer } }
            ]
        });

        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.imageProcessPipeline);
        computePass.setBindGroup(0, bindGroup);
        computePass.dispatchWorkgroups(64, 64);
        computePass.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }

    updateSimParams() {
        const params = new ArrayBuffer(48);
        const view = new DataView(params);

        view.setUint32(0, this.numDots, true);
        view.setFloat32(4, this.dotCharge, true);
        view.setFloat32(8, this.blankLevel, true);
        view.setFloat32(12, this.timeStep, true);
        view.setFloat32(16, this.maxVelocity, true);
        view.setFloat32(20, this.maxDisplacement, true);
        view.setFloat32(24, this.sustain, true);
        view.setUint32(28, this.imageWidth, true);
        view.setUint32(32, this.imageHeight, true);
        view.setFloat32(36, this.time, true);
        view.setFloat32(40, this.bmpCharge, true);
        view.setFloat32(44, this.aspectRatio, true);

        if (!this.simParamsBuffer) {
            this.simParamsBuffer = this.device.createBuffer({
                size: params.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
        }

        this.device.queue.writeBuffer(this.simParamsBuffer, 0, params);
    }

    createPipelines() {
        const computeShader = this.device.createShaderModule({
            code: COMPUTE_SHADER
        });

        const computeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' }
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: 'unfilterable-float' }
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    sampler: { type: 'non-filtering' }
                }
            ]
        });

        const computePipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [computeBindGroupLayout]
        });

        this.accelerationPipeline = this.device.createComputePipeline({
            layout: computePipelineLayout,
            compute: {
                module: computeShader,
                entryPoint: 'update_acceleration'
            }
        });

        this.velocityPipeline = this.device.createComputePipeline({
            layout: computePipelineLayout,
            compute: {
                module: computeShader,
                entryPoint: 'update_velocity'
            }
        });

        this.positionPipeline = this.device.createComputePipeline({
            layout: computePipelineLayout,
            compute: {
                module: computeShader,
                entryPoint: 'update_position'
            }
        });

        this.computeBindGroupLayout = computeBindGroupLayout;

        const renderShader = this.device.createShaderModule({
            code: RENDER_SHADER
        });

        this.renderParamsBuffer = this.device.createBuffer({
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: renderShader,
                entryPoint: 'vs_main'
            },
            fragment: {
                module: renderShader,
                entryPoint: 'fs_main',
                targets: [{
                    format: this.format
                }]
            },
            primitive: {
                topology: 'triangle-list'
            }
        });

        this.updateComputeBindGroup();
    }

    updateComputeBindGroup() {
        if (!this.computeBindGroupLayout) return;

        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.simParamsBuffer } },
                { binding: 1, resource: { buffer: this.positionBuffer } },
                { binding: 2, resource: { buffer: this.velocityBuffer } },
                { binding: 3, resource: { buffer: this.accelerationBuffer } },
                { binding: 4, resource: this.fieldTexture.createView() },
                { binding: 5, resource: this.sampler }
            ]
        });
    }

    simulate() {
        if (!this.running) return;

        this.time += this.timeStep;
        this.updateSimParams();

        const commandEncoder = this.device.createCommandEncoder();

        let computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.accelerationPipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.numDots / 256));
        computePass.end();

        computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.velocityPipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.numDots / 256));
        computePass.end();

        computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.positionPipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.numDots / 256));
        computePass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    render() {
        const renderParams = new Float32Array([
            this.canvas.width,
            this.canvas.height,
            this.dotSize,
            this.aspectRatio,
            this.sizeModulation,
            0, 0, 0,  // padding
            0, 0, 0, 0  // more padding
        ]);

        this.device.queue.writeBuffer(this.renderParamsBuffer, 0, renderParams);

        const renderBindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.renderParamsBuffer } },
                { binding: 1, resource: { buffer: this.positionBuffer } },
                { binding: 2, resource: this.inputTexture.createView() },
                { binding: 3, resource: this.sampler }
            ]
        });

        const commandEncoder = this.device.createCommandEncoder();
        const textureView = this.ctx.getCurrentTexture().createView();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.8, g: 0.8, b: 0.8, a: 1 },
                loadOp: 'clear',
                storeOp: 'store'
            }]
        });

        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, renderBindGroup);
        renderPass.draw(6, this.numDots);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    animate() {
        this.simulate();
        this.render();
        requestAnimationFrame(() => this.animate());
    }

    updateDotCount(count) {
        this.numDots = count;
        this.initParticles();
        this.updateComputeBindGroup();
    }

    updateChargeScale(scale) {
        this.chargeScale = scale;
        this.processImageGPU();
    }

    async loadAnimatedGif(file) {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }

        const arrayBuffer = await file.arrayBuffer();

        const gif = gifuct.parseGIF(arrayBuffer);
        const frames = gifuct.decompressFrames(gif, true);

        if (frames.length === 0) {
            console.error('No frames found in GIF');
            return;
        }

        const gifWidth = gif.lsd.width;
        const gifHeight = gif.lsd.height;

        const targetHeight = 243;
        const scale = targetHeight / gifHeight;

        const width = Math.floor(gifWidth * scale);
        const height = targetHeight;

        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });

        let currentFrame = 0;
        let lastFrameTime = performance.now();

        const animate = () => {
            const now = performance.now();
            const frame = frames[currentFrame];

            if (now - lastFrameTime >= (frame.delay || 100)) {
                if (frame.disposalType === 2) {
                    ctx.clearRect(0, 0, width, height);
                }

                const frameImageData = new ImageData(frame.patch, frame.dims.width, frame.dims.height);

                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = frame.dims.width;
                tempCanvas.height = frame.dims.height;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.putImageData(frameImageData, 0, 0);

                ctx.drawImage(tempCanvas,
                    frame.dims.left * scale,
                    frame.dims.top * scale,
                    frame.dims.width * scale,
                    frame.dims.height * scale
                );

                const fullImageData = ctx.getImageData(0, 0, width, height);

                this.updateGifTexture(fullImageData);

                currentFrame = (currentFrame + 1) % frames.length;
                lastFrameTime = now;
            }

            this.animationFrame = requestAnimationFrame(animate);
        };

        animate();
    }

    updateGifTexture(imageData) {
        if (this.imageWidth !== imageData.width || this.imageHeight !== imageData.height) {
            this.imageWidth = imageData.width;
            this.imageHeight = imageData.height;
            this.aspectRatio = this.imageWidth / this.imageHeight;

            this.canvas.width = this.imageWidth;
            this.canvas.height = this.imageHeight;

            this.createTextures();
            this.updateComputeBindGroup();
        }

        const rgbaData = new Uint8Array(imageData.data);

        let bmpChargeTotal = 0;
        for (let i = 0; i < rgbaData.length; i += 4) {
            const gray = (0.2989 * rgbaData[i] + 0.5870 * rgbaData[i + 1] + 0.1140 * rgbaData[i + 2]) / 255.0;
            bmpChargeTotal += this.blankLevel - gray;
        }

        const dotChargeTotal = this.numDots * this.dotCharge;
        this.bmpCharge = bmpChargeTotal > 0 ? dotChargeTotal / bmpChargeTotal : 1.0;

        this.imageData = rgbaData;

        if (this.device && this.inputTexture) {
            this.uploadImage();
            this.processImageGPU();
            this.updateSimParams();
        }
    }

}

async function initApp() {
    if (!navigator.gpu) {
        document.body.innerHTML = `<div style="font-family: 'Josefin Sans', sans-serif; color: #6A6A6A; text-align: center; margin-top: 100px; font-size: 14px;">webgpu is required but not available. try chrome, edge, or safari technology preview.</div>`;
        return;
    }

    const canvas = document.getElementById('canvas');
    const errorDiv = document.getElementById('error');

    try {
        const app = new BlueNoiseStippling(canvas);

        const response = await fetch('dots.gif');
        const blob = await response.blob();
        const file = new File([blob], 'dots.gif', { type: 'image/gif' });
        await app.loadAnimatedGif(file);

        const dotsSlider = document.getElementById('dotsSlider');
        const dotsValue = document.getElementById('dotsValue');
        dotsSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            dotsValue.textContent = value;
            app.updateDotCount(value);
        });

        const sizeSlider = document.getElementById('sizeSlider');
        const sizeValue = document.getElementById('sizeValue');
        sizeSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            sizeValue.textContent = value;
            app.dotSize = value;
        });

        const chargeSlider = document.getElementById('chargeSlider');
        const chargeValue = document.getElementById('chargeValue');
        chargeSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value) / 100;
            chargeValue.textContent = value.toFixed(2);
            app.updateChargeScale(value);
        });

        const modulationSlider = document.getElementById('modulationSlider');
        const modulationValue = document.getElementById('modulationValue');
        modulationSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            modulationValue.textContent = value + '%';
            app.sizeModulation = value / 100;
        });

        const fileInput = document.getElementById('fileInput');
        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file && file.type.startsWith('image/')) {
                const sourceLink = document.getElementById('sourceLink');
                if (sourceLink) sourceLink.style.display = 'none';
                if (file.type === 'image/gif') {
                    app.loadAnimatedGif(file);
                } else {
                    if (app.animationFrame) {
                        cancelAnimationFrame(app.animationFrame);
                    }

                    const img = new Image();
                    const url = URL.createObjectURL(file);

                    img.onload = () => {
                        const canvas = document.createElement('canvas');
                        const targetHeight = 243;

                        const scale = targetHeight / img.height;
                        const width = Math.floor(img.width * scale);
                        const height = targetHeight;

                        canvas.width = width;
                        canvas.height = height;

                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(img, 0, 0, width, height);

                        const imageData = ctx.getImageData(0, 0, width, height);
                        app.setImageData(imageData);

                        URL.revokeObjectURL(url);
                    };

                    img.src = url;
                }
            }
        });

    } catch (error) {
        console.error(error);
        let errorMessage = 'webgpu is not supported in your browser.';
        if (error.message.includes('WebGPU not supported')) {
            errorMessage = 'webgpu is required but not available. try chrome or something.';
        } else if (error.message.includes('No WebGPU adapter')) {
            errorMessage = 'no webgpu adapter found. check your gpu drivers.';
        }

        document.body.innerHTML = `<div style="font-family: 'Josefin Sans', sans-serif; color: #6A6A6A; text-align: center; margin-top: 100px; font-size: 14px;">${errorMessage}</div>`;
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}
