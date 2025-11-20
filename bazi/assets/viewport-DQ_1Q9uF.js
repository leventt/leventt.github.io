import{_ as ut}from"./index-DqbMt4OE.js";const dt={nearPlane:.01,farPlane:1e3,projectionType:"perspective",orthoSize:10,sensorWidth:36,sensorHeight:24,focalLength:35,principalPointX:.5,principalPointY:.5,radialK1:0,radialK2:0,radialK3:0,tangentialP1:0,tangentialP2:0,aspectRatioMode:"auto",fixedAspectRatio:16/9,defaultTheta:Math.PI*.25,defaultPhi:Math.PI*.25,defaultDistance:3,minDistance:.1,maxDistance:1e4,minPhi:.01,maxPhi:Math.PI-.01,defaultTarget:[0,0,0],rotationSpeed:.01,panSpeed:.01,zoomSpeed:.001,rotationDamping:.1,panDamping:.1,zoomDamping:.1},ht={voxelSize:.042,hashTableSize:4194304,maxVoxels:16777216},pt={raymarching:{maxSteps:96,surfaceEpsilon:.005,brushPreviewEpsilon:.001,adaptive:{enabled:!0,qualityMultiplier:1,minStepMultiplier:.1,maxStepMultiplier:2e3,complexityThreshold:.5,distanceScale:.1}}},ft={surfaceEpsilon:.1,preciseSurfaceEpsilon:.1,boundarySurfaceEpsilon:.1,raycastSurfaceEpsilon:.1,absoluteSurfaceEpsilon:.001,absoluteBrushEpsilon:.001,absoluteUpdateThreshold:.001,emptySpaceMultiplier:2,largeEmptySpaceMultiplier:1.3,surfaceThresholdMultiplier:1,brushPreviewThresholdMultiplier:1,maxSDFMultiplier:1.3},Ge={alwaysCalculateGBufferNormals:!1,input:{defaultTabletPressure:.5,borderThresholdPercent:.1,mouseThrottleFPS:13},adaptiveCache:{maxCacheSize:256,maxSculptingDistance:25},brushes:{erode:{strengthBias:.15}}},Ie={bufferSizes:{brushParameters:64,raycastParameters:64,raycastResult:64,sculptingUniforms:32,CACHE_ELEMENT_SIZE:4,CACHE_METADATA_SIZE:64,MIN_DUMMY_CACHE_SIZE:16}},d={voxel:ht,bufferSizes:Ie.bufferSizes,camera:dt,sculpting:Ge,raymarching:pt.raymarching,gpu:Ie,precision:ft},De=`
// PRECISION CONSTANTS

// surface detection epsilon values (multipliers of voxelSize)
const SURFACE_EPSILON = ${d.precision.surfaceEpsilon}f;
const PRECISE_SURFACE_EPSILON = ${d.precision.preciseSurfaceEpsilon}f;
const BOUNDARY_SURFACE_EPSILON = ${d.precision.boundarySurfaceEpsilon}f;
const RAYCAST_SURFACE_EPSILON = ${d.precision.raycastSurfaceEpsilon}f;

// absolute epsilon values (not multiplied by voxelSize but absolute values)
const ABSOLUTE_SURFACE_EPSILON = ${d.precision.absoluteSurfaceEpsilon}f;
const ABSOLUTE_BRUSH_EPSILON = ${d.precision.absoluteBrushEpsilon}f;
const ABSOLUTE_UPDATE_THRESHOLD = ${d.precision.absoluteUpdateThreshold}f;

// empty space multipliers
const EMPTY_SPACE_MULTIPLIER = ${d.precision.emptySpaceMultiplier}f;
const LARGE_EMPTY_SPACE_MULTIPLIER = ${d.precision.largeEmptySpaceMultiplier}f;

// surface threshold multipliers
const SURFACE_THRESHOLD_MULTIPLIER = ${d.precision.surfaceThresholdMultiplier}f;
const BRUSH_PREVIEW_THRESHOLD_MULTIPLIER = ${d.precision.brushPreviewThresholdMultiplier}f;

// SDF value bounds
const MAX_SDF_MULTIPLIER = ${d.precision.maxSDFMultiplier}f;

// tetrahedron constants for normal calculation (Inigo Quilez method)
const TETRAHEDRON_K1 = vec3<f32>( 1.0, -1.0, -1.0);
const TETRAHEDRON_K2 = vec3<f32>(-1.0, -1.0,  1.0);
const TETRAHEDRON_K3 = vec3<f32>(-1.0,  1.0, -1.0);
const TETRAHEDRON_K4 = vec3<f32>( 1.0,  1.0,  1.0);


// CORE STRUCTURES


// structure for voxel data - SDF only
struct VoxelData {
    sdf: f32
}

struct BrushParams {
    position: vec3<f32>,
    radius: f32,
    strength: f32,
    operation: u32,       // 0=move, 1=bump, 2=erode
    falloff_type: u32,    // 0=constant, 1=linear, 2=smooth, 3=gaussian, 4=sharp
    target_value: f32,    // For flatten/smooth operations
    normal: vec3<f32>,
    _padding: u32         // Align to 16-byte boundary
}


// PRECISION HELPER FUNCTIONS

// helper functions for consistent epsilon calculation
fn get_surface_epsilon(voxel_size: f32) -> f32 {
    return voxel_size * SURFACE_EPSILON;
}

fn get_precise_surface_epsilon(voxel_size: f32) -> f32 {
    return voxel_size * PRECISE_SURFACE_EPSILON;
}

fn get_boundary_surface_epsilon(voxel_size: f32) -> f32 {
    return voxel_size * BOUNDARY_SURFACE_EPSILON;
}

fn get_raycast_surface_epsilon(voxel_size: f32) -> f32 {
    return voxel_size * RAYCAST_SURFACE_EPSILON;
}


fn get_surface_threshold(voxel_size: f32) -> f32 {
    return voxel_size * SURFACE_THRESHOLD_MULTIPLIER;
}

fn get_brush_preview_threshold(voxel_size: f32) -> f32 {
    return voxel_size * BRUSH_PREVIEW_THRESHOLD_MULTIPLIER;
}



// COORDINATE UTILITIES

// get voxel index using explicit size (preferred method)
fn get_voxel_index_with_size(voxel_pos: vec3<i32>, size: i32) -> u32 {
    return u32(voxel_pos.z * size * size + voxel_pos.y * size + voxel_pos.x);
}


// Helper to calculate probe distance with wraparound
fn probe_distance(current_pos: u32, ideal_pos: u32, table_size: u32) -> u32 {
    if (current_pos >= ideal_pos) {
        return current_pos - ideal_pos;
    } else {  // Added 'else' for clarity
        return current_pos + table_size - ideal_pos;
    }
}

// ================================
// UTILITY FUNCTIONS
// ================================
// Common utility functions for shader operations
fn saturate(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

`,mt=`
// Constants for voxel hash map
const HASH_TABLE_SIZE: u32 = 4194304u;  // Must match VoxelHashMap.js - 4M entries
const EMPTY_HASH_SLOT: i32 = -1;
const EMPTY_COORD_MARKER: i32 = 2147483647; // INT32_MAX to mark empty slots
const MAX_HASH_PROBES: u32 = 32u;      // Reduced from 256 - most empty slots found in 1-2 probes, saves 87% of worst-case hash lookups
const VOXEL_EMPTY_SDF: f32 = 10.0;   // Empty space value - increased for better ray marching performance

// Hash function for 3D coordinates - MurmurHash3-like implementation
fn hash_coords(x: i32, y: i32, z: i32) -> u32 {
    let seed = 0x9747b28cu;
    
    // Mix coordinates separately to avoid bit overflow
    // Each coordinate gets its own mixing
    var h1 = seed ^ u32(x);
    h1 = h1 * 0xcc9e2d51u;
    h1 = (h1 << 15u) | (h1 >> 17u);
    h1 = h1 * 0x1b873593u;
    
    var h2 = seed ^ u32(y);
    h2 = h2 * 0xcc9e2d51u;
    h2 = (h2 << 15u) | (h2 >> 17u);
    h2 = h2 * 0x1b873593u;
    
    var h3 = seed ^ u32(z);
    h3 = h3 * 0xcc9e2d51u;
    h3 = (h3 << 15u) | (h3 >> 17u);
    h3 = h3 * 0x1b873593u;
    
    // Combine and final mix
    var h = seed;
    h = h ^ h1;
    h = ((h << 13u) | (h >> 19u)) + 0xe6546b64u;
    h = h ^ h2;
    h = ((h << 13u) | (h >> 19u)) + 0xe6546b64u;
    h = h ^ h3;
    
    // Final avalanche
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    
    return h % HASH_TABLE_SIZE;
}

// World position to voxel coordinates - CENTER-BASED SYSTEM
fn world_to_voxel_coord(world_pos: vec3<f32>, voxel_size: f32) -> vec3<i32> {
    // CENTER-BASED: voxel (i,j,k) contains SDF sampled at center (i+0.5, j+0.5, k+0.5) * voxel_size
    return vec3<i32>(floor(world_pos / voxel_size));
}

// Voxel coordinates to world center position - CENTER-BASED SYSTEM
fn voxel_coord_to_world_center(voxel_coord: vec3<i32>, voxel_size: f32) -> vec3<f32> {
    // Return the center position where SDF is sampled and stored
    return (vec3<f32>(voxel_coord) + 0.5) * voxel_size;
}

// Find voxel in hash table using linear probing
fn find_voxel_slot(voxel_coord: vec3<i32>, hash_table: ptr<storage, array<vec4<i32>>, read>) -> i32 {
    let base_hash = hash_coords(voxel_coord.x, voxel_coord.y, voxel_coord.z);
    
    // Linear probing to find the voxel
    for (var i = 0u; i < MAX_HASH_PROBES; i++) {
        let slot = (base_hash + i) % HASH_TABLE_SIZE;
        let entry = (*hash_table)[slot];
        
        // Check if this is an empty slot
        if (entry.w == EMPTY_HASH_SLOT || entry.x == EMPTY_COORD_MARKER) {
            return EMPTY_HASH_SLOT;
        }
        
        // Check if this slot contains our voxel
        if (entry.x == voxel_coord.x && entry.y == voxel_coord.y && entry.z == voxel_coord.z) {
            return entry.w; // Return voxel data index
        }
    }
    
    // Hash table full or max probes reached
    return EMPTY_HASH_SLOT;
}

// Query SDF value at world position
fn query_voxel_sdf(world_pos: vec3<f32>, voxel_size: f32, hash_table: ptr<storage, array<vec4<i32>>, read>, voxel_data: ptr<storage, array<f32>, read>) -> f32 {
    let voxel_coord = world_to_voxel_coord(world_pos, voxel_size);
    let voxel_index = find_voxel_slot(voxel_coord, hash_table);
    
    if (voxel_index == EMPTY_HASH_SLOT) {
        return VOXEL_EMPTY_SDF;
    }
    
    // Bounds check
    if (voxel_index < 0) {
        return VOXEL_EMPTY_SDF;
    }
    
    // Additional safety check for array bounds
    let array_length = i32(arrayLength(voxel_data));
    if (voxel_index >= array_length) {
        return VOXEL_EMPTY_SDF;
    }
    
    let raw_sdf = (*voxel_data)[voxel_index];
    return clamp(raw_sdf, -10.0, 10.0);
}

// Trilinear interpolation for smooth SDF - CENTER-BASED SYSTEM
fn query_voxel_sdf_trilinear(world_pos: vec3<f32>, voxel_size: f32, hash_table: ptr<storage, array<vec4<i32>>, read>, voxel_data: ptr<storage, array<f32>, read>) -> f32 {
    // CENTER-BASED: interpolate between 8 voxel centers surrounding the query point
    
    // Find the interpolation cube - base voxel for the 8-corner interpolation
    let base_voxel = vec3<i32>(floor(world_pos / voxel_size - 0.5));
    
    // Calculate interpolation weights [0,1] within the voxel cube
    let base_world_center = voxel_coord_to_world_center(base_voxel, voxel_size);
    let local_pos = (world_pos - base_world_center) / voxel_size;
    
    // Ensure weights are properly clamped to avoid artifacts
    let weights = clamp(local_pos, vec3<f32>(0.0), vec3<f32>(1.0));
    
    // Sample 8 corners
    var values: array<f32, 8>;
    var valid_mask: array<bool, 8>;
    var has_surface = false;
    
    for (var i = 0u; i < 8u; i++) {
        let offset = vec3<i32>(
            i32((i >> 0u) & 1u),
            i32((i >> 1u) & 1u),
            i32((i >> 2u) & 1u)
        );
        let sample_coord = base_voxel + offset;
        let voxel_index = find_voxel_slot(sample_coord, hash_table);
        
        if (voxel_index == EMPTY_HASH_SLOT) {
            values[i] = VOXEL_EMPTY_SDF;
            valid_mask[i] = false;
        } else {
            // Bounds check
            if (voxel_index < 0) {
                values[i] = VOXEL_EMPTY_SDF;
                valid_mask[i] = false;
            } else {
                let array_length = i32(arrayLength(voxel_data));
                if (voxel_index >= array_length) {
                    values[i] = VOXEL_EMPTY_SDF;
                    valid_mask[i] = false;
                } else {
                    let raw_sdf = (*voxel_data)[voxel_index];
                    values[i] = clamp(raw_sdf, -10.0, 10.0);
                    valid_mask[i] = true;
                    // Check if we're near a surface
                    if (abs(raw_sdf) < voxel_size * 2.0) {
                        has_surface = true;
                    }
                }
            }
        }
    }
    
    // Count valid corners
    var valid_count = 0u;
    for (var i = 0u; i < 8u; i++) {
        if (valid_mask[i]) {
            valid_count++;
        }
    }
    
    // CRITICAL FIX: Never interpolate with empty space markers
    // This prevents contamination of real SDF values with empty markers
    if (valid_count == 0u) {
        // No valid samples - return empty
        return VOXEL_EMPTY_SDF;
    } else if (valid_count < 8u) {
        // Mixed case - cannot safely interpolate with empty markers
        // Use nearest valid sample or average of valid samples only
        var sum = 0.0;
        var closest_dist = 999999.0;
        var closest_value = VOXEL_EMPTY_SDF;
        
        for (var i = 0u; i < 8u; i++) {
            if (valid_mask[i]) {
                sum += values[i];
                
                // Calculate distance to this corner for nearest neighbor
                let corner_offset = vec3<f32>(
                    f32((i >> 0u) & 1u),
                    f32((i >> 1u) & 1u),
                    f32((i >> 2u) & 1u)
                );
                let dist = distance(weights, corner_offset);
                if (dist < closest_dist) {
                    closest_dist = dist;
                    closest_value = values[i];
                }
            }
        }
        
        // For surface-near queries, use nearest valid sample
        // For far queries, use average of valid samples
        if (has_surface && valid_count <= 4u) {
            return closest_value;
        } else {
            return sum / f32(valid_count);
        }
    } else {
        // All corners valid - safe to interpolate
        let x0 = mix(values[0], values[1], weights.x);
        let x1 = mix(values[2], values[3], weights.x);
        let x2 = mix(values[4], values[5], weights.x);
        let x3 = mix(values[6], values[7], weights.x);
        
        let y0 = mix(x0, x1, weights.y);
        let y1 = mix(x2, x3, weights.y);
        
        return mix(y0, y1, weights.z);
    }
}

// Calculate gradient using finite differences - CENTER-BASED SYSTEM
fn calculate_sdf_gradient(world_pos: vec3<f32>, voxel_size: f32, hash_table: ptr<storage, array<vec4<i32>>, read>, voxel_data: ptr<storage, array<f32>, read>) -> vec3<f32> {
    // CENTER-BASED: gradient calculation using center-sampled SDF values
    let center_value = query_voxel_sdf_trilinear(world_pos, voxel_size, hash_table, voxel_data);
    
    // Adaptive step size: smaller near surface, larger far away
    let h = max(voxel_size * 0.5, min(voxel_size * 2.0, abs(center_value) * 0.1));
    
    // Use central differences for better accuracy
    let dx = query_voxel_sdf_trilinear(world_pos + vec3<f32>(h, 0.0, 0.0), voxel_size, hash_table, voxel_data) - 
             query_voxel_sdf_trilinear(world_pos - vec3<f32>(h, 0.0, 0.0), voxel_size, hash_table, voxel_data);
    let dy = query_voxel_sdf_trilinear(world_pos + vec3<f32>(0.0, h, 0.0), voxel_size, hash_table, voxel_data) - 
             query_voxel_sdf_trilinear(world_pos - vec3<f32>(0.0, h, 0.0), voxel_size, hash_table, voxel_data);
    let dz = query_voxel_sdf_trilinear(world_pos + vec3<f32>(0.0, 0.0, h), voxel_size, hash_table, voxel_data) - 
             query_voxel_sdf_trilinear(world_pos - vec3<f32>(0.0, 0.0, h), voxel_size, hash_table, voxel_data);
    
    // Normalize gradient (central differences use 2h)
    var gradient = vec3<f32>(dx, dy, dz) / (2.0 * h);
    
    // Ensure we have a valid normal
    let len = length(gradient);
    if (len < 0.0001) {
        // If gradient is too small, try with larger step
        let h2 = voxel_size * 2.0;
        let dx2 = query_voxel_sdf(world_pos + vec3<f32>(h2, 0.0, 0.0), voxel_size, hash_table, voxel_data) - 
                  query_voxel_sdf(world_pos - vec3<f32>(h2, 0.0, 0.0), voxel_size, hash_table, voxel_data);
        let dy2 = query_voxel_sdf(world_pos + vec3<f32>(0.0, h2, 0.0), voxel_size, hash_table, voxel_data) - 
                  query_voxel_sdf(world_pos - vec3<f32>(0.0, h2, 0.0), voxel_size, hash_table, voxel_data);
        let dz2 = query_voxel_sdf(world_pos + vec3<f32>(0.0, 0.0, h2), voxel_size, hash_table, voxel_data) - 
                  query_voxel_sdf(world_pos - vec3<f32>(0.0, 0.0, h2), voxel_size, hash_table, voxel_data);
        gradient = vec3<f32>(dx2, dy2, dz2) / (2.0 * h2);
        let len2 = length(gradient);
        if (len2 < 0.0001) {
            return vec3<f32>(0.0, 1.0, 0.0); // Default up vector
        }
        return gradient / len2;
    }
    return gradient / len;
}

// Adaptive gradient calculation based on viewing distance
fn calculate_sdf_gradient_adaptive(world_pos: vec3<f32>, voxel_size: f32, view_distance: f32, pixel_cone_angle: f32, hash_table: ptr<storage, array<vec4<i32>>, read>, voxel_data: ptr<storage, array<f32>, read>) -> vec3<f32> {
    // Compute cone width at this distance
    let cone_width = max(view_distance * pixel_cone_angle, voxel_size);
    
    // Adaptive step size based on cone width
    let h = max(cone_width * 0.25, voxel_size * 0.5);
    
    // Use central differences
    let dx = query_voxel_sdf_trilinear(world_pos + vec3<f32>(h, 0.0, 0.0), voxel_size, hash_table, voxel_data) - 
             query_voxel_sdf_trilinear(world_pos - vec3<f32>(h, 0.0, 0.0), voxel_size, hash_table, voxel_data);
    let dy = query_voxel_sdf_trilinear(world_pos + vec3<f32>(0.0, h, 0.0), voxel_size, hash_table, voxel_data) - 
             query_voxel_sdf_trilinear(world_pos - vec3<f32>(0.0, h, 0.0), voxel_size, hash_table, voxel_data);
    let dz = query_voxel_sdf_trilinear(world_pos + vec3<f32>(0.0, 0.0, h), voxel_size, hash_table, voxel_data) - 
             query_voxel_sdf_trilinear(world_pos - vec3<f32>(0.0, 0.0, h), voxel_size, hash_table, voxel_data);
    
    // Normalize gradient
    var gradient = vec3<f32>(dx, dy, dz) / (2.0 * h);
    let len = length(gradient);
    
    if (len < 0.0001) {
        // Fallback to standard calculation
        return calculate_sdf_gradient(world_pos, voxel_size, hash_table, voxel_data);
    }
    
    return gradient / len;
}

// Parameters for voxel hash map
struct VoxelHashMapParams {
    voxel_size: f32,
    hash_table_size: u32,
    max_voxels: u32,
    voxel_count: u32,
    block_size: u32,
    occupancy_table_size: u32,
    padding1: u32,
    padding2: u32
}
`,Pe=`
${mt}

// Main query function using VoxelHashMap - parameters passed from caller
fn query_voxel_hashmap(world_pos: vec3<f32>, voxel_size: f32, hash_table: ptr<storage, array<vec4<i32>>, read>, sdf_data: ptr<storage, array<f32>, read>) -> f32 {
    return query_voxel_sdf_trilinear(
        world_pos,
        voxel_size,
        hash_table,
        sdf_data
    );
}

// Calculate normal using gradient
fn calculate_voxel_normal(world_pos: vec3<f32>, voxel_size: f32, hash_table: ptr<storage, array<vec4<i32>>, read>, sdf_data: ptr<storage, array<f32>, read>) -> vec3<f32> {
    return calculate_sdf_gradient(
        world_pos,
        voxel_size,
        hash_table,
        sdf_data
    );
}
`,We=`
// ================================
// RAY TRACING STRUCTURES
// ================================
// Ray intersection result
struct RayHit {
    hit: bool,
    distance: f32,
    position: vec3<f32>,
    normal: vec3<f32>,
}

// DDA state for hierarchical traversal
struct DDAState {
    t_max: vec3<f32>,      // next intersection times for each axis
    t_delta: vec3<f32>,    // step size for each axis
    step: vec3<i32>,       // step direction (-1 or 1)
    current_pos: vec3<i32>, // current voxel/brick position
}

// ================================
// RENDERING UTILITIES
// ================================
// Fullscreen triangle vertex generation
// Creates a triangle that covers the entire screen when rendered
// Usage: @vertex fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4f { return fullscreen_triangle_vertex(vertex_index); }
fn fullscreen_triangle_vertex(vertex_index: u32) -> vec4f {
    // This creates a single triangle that covers the entire screen
    // vertex_index: 0, 1, 2 map to the three corners of the triangle
    let x = f32((vertex_index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vertex_index & 2u) * 2.0 - 1.0;
    return vec4f(x, y, 0.0, 1.0);
}

// Convert fragment coordinates to normalized UV coordinates [0, 1]
fn frag_coord_to_uv(frag_coord: vec2f, resolution: vec2f) -> vec2f {
    return frag_coord / resolution;
}

// Convert fragment coordinates to normalized device coordinates [-1, 1]
fn frag_coord_to_ndc(frag_coord: vec2f, resolution: vec2f) -> vec2f {
    let uv = frag_coord_to_uv(frag_coord, resolution);
    return uv * 2.0 - 1.0;
}

// Convert UV coordinates [0, 1] to normalized device coordinates [-1, 1]
fn uv_to_ndc(uv: vec2f) -> vec2f {
    return uv * 2.0 - 1.0;
}

// Convert normalized device coordinates [-1, 1] to UV coordinates [0, 1]
fn ndc_to_uv(ndc: vec2f) -> vec2f {
    return ndc * 0.5 + 0.5;
}

// Gamma correction
fn linear_to_srgb(color: vec3f) -> vec3f {
    return pow(color, vec3f(1.0 / 2.2));
}

fn srgb_to_linear(color: vec3f) -> vec3f {
    return pow(color, vec3f(2.2));
}

// ================================
// HYBRID DDA TRAVERSAL
// ================================
// Initialize DDA state for a given grid scale
fn init_dda(origin: vec3<f32>, direction: vec3<f32>, grid_size: f32) -> DDAState {
    var state: DDAState;

    // Calculate which grid cell we're starting in
    state.current_pos = vec3<i32>(floor(origin / grid_size));

    // Calculate step direction for each axis
    state.step = vec3<i32>(sign(direction));

    // Calculate t_delta - how far we travel along ray for each grid step
    let inv_dir = 1.0 / (direction + vec3<f32>(1e-8)); // avoid division by zero
    state.t_delta = abs(grid_size * inv_dir);

    // Calculate t_max - distance to next grid boundary
    var next_boundary = vec3<f32>(state.current_pos) * grid_size;
    if (state.step.x > 0) { next_boundary.x += grid_size; }
    if (state.step.y > 0) { next_boundary.y += grid_size; }
    if (state.step.z > 0) { next_boundary.z += grid_size; }

    state.t_max = (next_boundary - origin) * inv_dir;

    // Handle rays parallel to axes
    if (abs(direction.x) < 1e-8) { state.t_max.x = 1e10; }
    if (abs(direction.y) < 1e-8) { state.t_max.y = 1e10; }
    if (abs(direction.z) < 1e-8) { state.t_max.z = 1e10; }

    return state;
}

// Step DDA to next cell, returns distance traveled
fn step_dda(state: ptr<function, DDAState>) -> f32 {
    let prev_t = min(min((*state).t_max.x, (*state).t_max.y), (*state).t_max.z);

    // Step along the axis with smallest t_max
    if ((*state).t_max.x < (*state).t_max.y && (*state).t_max.x < (*state).t_max.z) {
        (*state).current_pos.x += (*state).step.x;
        (*state).t_max.x += (*state).t_delta.x;
    } else if ((*state).t_max.y < (*state).t_max.z) {
        (*state).current_pos.y += (*state).step.y;
        (*state).t_max.y += (*state).t_delta.y;
    } else {
        (*state).current_pos.z += (*state).step.z;
        (*state).t_max.z += (*state).t_delta.z;
    }

    return prev_t;
}


// ================================
// ADAPTIVE STEPPING & COMPLEXITY
// ================================

// Adaptive step size calculation based on distance and scene complexity
fn get_adaptive_step_size(distance: f32, complexity: f32, surface_epsilon: f32, adaptive_distance_scale: f32, adaptive_max_step_multiplier: f32, adaptive_quality_multiplier: f32, adaptive_min_step_multiplier: f32) -> f32 {
    let base_step = surface_epsilon;
    let distance_scale = min(distance * adaptive_distance_scale, adaptive_max_step_multiplier);
    let complexity_scale = mix(2.0, 0.5, complexity);
    
    // Start with distance-based step, then scale by complexity
    let distance_based_step = max(distance * 0.7, base_step);
    let adaptive_step = distance_based_step * complexity_scale * adaptive_quality_multiplier;
    
    // Clamp to min/max bounds
    let min_step = base_step * adaptive_min_step_multiplier;
    let max_step = base_step * adaptive_max_step_multiplier;
    
    return clamp(adaptive_step, min_step, max_step);
}

// ================================
// UNIFIED SPATIAL RAY MARCHING
// ================================
// Requires: query_sdf(pos: vec3<f32>) -> f32 function
fn trace_ray(
    origin: vec3<f32>, 
    direction: vec3<f32>, 
    max_steps: i32, 
    max_distance: f32,
    voxel_size: f32
) -> RayHit {
    var result: RayHit;
    result.hit = false;
    result.distance = 0.0;
    
    var t = voxel_size * 0.05; // Start slightly off ray origin
    let hit_epsilon = voxel_size * 0.04; // Detection threshold
    
    // Distance-dependent step count: allocate more steps for distant rays
    // Rays reaching max_distance get max_steps, rays stopping early use fewer
    var step_count = 0;
    var avg_step_size = max_distance / f32(max_steps);
    
    for (var i = 0; i < max_steps; i++) {
        if (t >= max_distance) {
            break;
        }
        
        let pos = origin + direction * t;
        
        // Query SDF directly
        let d = query_sdf(pos);
        
        if (d < hit_epsilon) {
            result.hit = true;
            result.distance = t;
            result.position = pos;
            return result;
        }
        
        // Sphere marching: pure SDF-based stepping without distance scaling
        // Trust the SDF to guide stepping correctly at all distances
        let base_step = clamp(d * 1.0, voxel_size * 0.05, voxel_size * 1.5); // Trust SDF fully
        
        let step_size = base_step;
        t += step_size;
        step_count += 1;
    }
    
    return result;
}
`,gt=`
// extract camera basis vectors from view matrix
// expects column-major matrix storage (WebGPU/WGSL standard)
fn get_camera_basis(view_matrix: mat4x4<f32>) -> array<vec3<f32>, 3> {
    // in column-major storage:
    // - matrix[i] gets column i
    // - to get row vectors, we need to take element [i] from each column
    let right = vec3<f32>(view_matrix[0][0], view_matrix[1][0], view_matrix[2][0]);
    let up = vec3<f32>(view_matrix[0][1], view_matrix[1][1], view_matrix[2][1]);
    // negate forward because camera looks down -Z in right-handed system
    let forward = vec3<f32>(-view_matrix[0][2], -view_matrix[1][2], -view_matrix[2][2]);

    return array<vec3<f32>, 3>(right, up, forward);
}

// extract field of view from projection matrix
// works for perspective projection matrices
fn get_fov_from_projection(proj_matrix: mat4x4<f32>) -> f32 {
    // for perspective projection matrix:
    // proj_matrix[1][1] = 1 / tan(fov/2)
    // so: tan(fov/2) = 1 / proj_matrix[1][1]
    return 1.0 / proj_matrix[1][1];
}

// generate ray direction from normalized screen coordinates
// uv should be in range [-1, 1]
fn generate_ray_direction(
    uv: vec2<f32>,
    aspect_ratio: f32,
    tan_half_fov: f32,
    cam_right: vec3<f32>,
    cam_up: vec3<f32>,
    cam_forward: vec3<f32>
) -> vec3<f32> {
    return normalize(
        cam_right * uv.x * aspect_ratio * tan_half_fov +
        cam_up * uv.y * tan_half_fov +
        cam_forward
    );
}

// convert screen coordinates to normalized device coordinates
// frag_coord: pixel coordinates from fragment shader
// resolution: screen resolution in pixels
// returns UV in range [-1, 1] with correct aspect ratio
fn screen_to_ndc(frag_coord: vec2<f32>, resolution: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        (frag_coord.x / resolution.x) * 2.0 - 1.0,
        1.0 - (frag_coord.y / resolution.y) * 2.0  // flip Y
    );
}

// complete ray generation from fragment coordinates
// combines all the above functions for convenience
fn generate_ray_from_screen(
    frag_coord: vec2<f32>,
    resolution: vec2<f32>,
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>
) -> vec3<f32> {
    let uv = screen_to_ndc(frag_coord, resolution);
    let aspect_ratio = resolution.x / resolution.y;

    let cam_basis = get_camera_basis(view_matrix);
    let tan_half_fov = get_fov_from_projection(proj_matrix);

    return generate_ray_direction(
        uv, aspect_ratio, tan_half_fov,
        cam_basis[0], cam_basis[1], cam_basis[2]
    );
}
`,Xe=`
// Query adaptive cache for SDF value - CENTER-BASED SYSTEM
fn query_adaptive_cache(
    world_pos: vec3<f32>,
    cache_data: ptr<storage, array<f32>, read>,
    cache_origin: vec3<f32>,
    cache_dimensions: vec3<u32>,
    cache_voxel_size: f32
) -> f32 {
    // CENTER-BASED: Convert world position to cache local position
    let local_pos = world_pos - cache_origin;
    
    // ROBUST COORDINATE SYSTEM: Cache origin is corner, cache[i] sampled at origin + (i + 0.5) * voxel_size
    // To interpolate, find where world_pos falls relative to cache sample points
    let voxel_pos = (local_pos / cache_voxel_size) - 0.5;
    
    // For trilinear interpolation, voxel_pos should be >= 0 and < dimensions-1
    // This ensures we can safely access the 8 surrounding cache values
    if (any(voxel_pos < vec3<f32>(0.0)) || any(voxel_pos >= vec3<f32>(cache_dimensions) - vec3<f32>(1.0))) {
        return VOXEL_EMPTY_SDF; // Outside interpolation bounds
    }
    
    // Use voxel_pos directly for interpolation coordinates (no shift needed)
    let clamped_voxel_pos = clamp(voxel_pos, vec3<f32>(0.0), vec3<f32>(cache_dimensions) - vec3<f32>(1.0));
    
    // Trilinear interpolation for smooth results
    let voxel_coord = floor(clamped_voxel_pos);
    let fract_pos = clamped_voxel_pos - voxel_coord;
    
    // Get integer coordinates
    let x0 = u32(voxel_coord.x);
    let y0 = u32(voxel_coord.y);
    let z0 = u32(voxel_coord.z);
    
    let x1 = min(x0 + 1u, cache_dimensions.x - 1u);
    let y1 = min(y0 + 1u, cache_dimensions.y - 1u);
    let z1 = min(z0 + 1u, cache_dimensions.z - 1u);
    
    // Sample 8 corners
    let v000 = (*cache_data)[x0 + y0 * cache_dimensions.x + z0 * cache_dimensions.x * cache_dimensions.y];
    let v100 = (*cache_data)[x1 + y0 * cache_dimensions.x + z0 * cache_dimensions.x * cache_dimensions.y];
    let v010 = (*cache_data)[x0 + y1 * cache_dimensions.x + z0 * cache_dimensions.x * cache_dimensions.y];
    let v110 = (*cache_data)[x1 + y1 * cache_dimensions.x + z0 * cache_dimensions.x * cache_dimensions.y];
    let v001 = (*cache_data)[x0 + y0 * cache_dimensions.x + z1 * cache_dimensions.x * cache_dimensions.y];
    let v101 = (*cache_data)[x1 + y0 * cache_dimensions.x + z1 * cache_dimensions.x * cache_dimensions.y];
    let v011 = (*cache_data)[x0 + y1 * cache_dimensions.x + z1 * cache_dimensions.x * cache_dimensions.y];
    let v111 = (*cache_data)[x1 + y1 * cache_dimensions.x + z1 * cache_dimensions.x * cache_dimensions.y];
    
    // Trilinear interpolation
    let x00 = mix(v000, v100, fract_pos.x);
    let x10 = mix(v010, v110, fract_pos.x);
    let x01 = mix(v001, v101, fract_pos.x);
    let x11 = mix(v011, v111, fract_pos.x);
    
    let y_0 = mix(x00, x10, fract_pos.y);
    let y_1 = mix(x01, x11, fract_pos.y);
    
    return mix(y_0, y_1, fract_pos.z);
}
`,U={metadata:`
// Cache metadata structure - used by ALL shaders (32 bytes + padding)
struct CacheMetadata {
    origin: vec3<f32>,      // offset 0  (12 bytes)
    voxel_size: f32,        // offset 12 (4 bytes)
    dimensions: vec3<u32>,  // offset 16 (12 bytes)
    is_active: u32,         // offset 28 (4 bytes)
    // Total: 32 bytes, padded to 64 for uniform alignment
}`,constants:`
// Cache-related constants
const CACHE_EPSILON: f32 = 0.001;
const CACHE_MAX_SDF: f32 = 10.0;
const CACHE_MIN_SDF: f32 = -10.0;`,utilities:`
// Clamp SDF to reasonable range
fn clamp_sdf(sdf: f32) -> f32 {
    return clamp(sdf, CACHE_MIN_SDF, CACHE_MAX_SDF);
}

// Saturate function (clamp to 0-1)
fn saturate(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}`},He=`
${U.metadata}
${U.constants}
${De}
${Pe}
${gt}
${We}

// VoxelHashMap bindings - new system
@group(1) @binding(0) var<storage, read> voxel_hash_table: array<vec4<i32>>;  // Hash table entries
@group(1) @binding(1) var<storage, read> voxel_sdf_data: array<f32>;         // SDF values
@group(1) @binding(2) var<storage, read> voxel_params_buffer: VoxelHashMapParams;  // Hash map parameters

// Include adaptive cache query
${Xe}

// Adaptive Cache bindings
@group(2) @binding(0) var<storage, read> cache_data: array<f32>;
@group(2) @binding(1) var<uniform> cache_metadata: CacheMetadata;

// Depth distribution functions are imported from shared module


// Query SDF with adaptive cache integration and consistent precision
fn query_sdf(world_pos: vec3<f32>) -> f32 {
    // Check if adaptive cache is active
    if (cache_metadata.is_active > 0u) {
        // Try to sample from adaptive cache
        let cache_value = query_adaptive_cache(
            world_pos,
            &cache_data,
            cache_metadata.origin,
            cache_metadata.dimensions,
            cache_metadata.voxel_size
        );
        
        // CONSISTENCY FIX: Use consistent empty space threshold across all systems
        // This prevents surface detection inconsistencies between cache and voxel queries
        let empty_space_threshold = max(cache_metadata.voxel_size * 6.0, voxel_params_buffer.voxel_size * 6.0);
        if (cache_value < empty_space_threshold) {
            return cache_value;
        }
    }
    
    // Fallback to voxel hash map with consistent precision
    return query_voxel_hashmap(world_pos, voxel_params_buffer.voxel_size, &voxel_hash_table, &voxel_sdf_data);
}

// Get the effective voxel size for ray marching step calculations
fn get_effective_voxel_size() -> f32 {
    // Use cache voxel size if cache is active, otherwise use base voxel size
    if (cache_metadata.is_active > 0u) {
        return cache_metadata.voxel_size;
    }
    return voxel_params_buffer.voxel_size;
}

// Calculate normal using optimized tetrahedron sampling (4 SDF queries, high quality)
// More efficient than central differences (6 queries) while maintaining quality
fn calculate_normal(world_pos: vec3<f32>) -> vec3<f32> {
    let base_voxel_size = voxel_params_buffer.voxel_size;
    let cache_voxel_size = cache_metadata.voxel_size;
    
    // Use the finer resolution for more accurate normals
    let finest_voxel_size = select(base_voxel_size, cache_voxel_size, cache_metadata.is_active > 0u && cache_voxel_size < base_voxel_size);
    let eps = finest_voxel_size * 0.4; // Smaller epsilon for tetrahedron method
    
    // Tetrahedron sampling: 4 samples at tetrahedral positions
    // This gives smooth, isotropic normals with only 4 SDF queries
    let p0 = world_pos;
    let p1 = world_pos + vec3<f32>(eps, eps, eps);
    let p2 = world_pos + vec3<f32>(eps, -eps, -eps);
    let p3 = world_pos + vec3<f32>(-eps, eps, -eps);
    let p4 = world_pos + vec3<f32>(-eps, -eps, eps);
    
    let d0 = query_sdf(p0);
    let d1 = query_sdf(p1);
    let d2 = query_sdf(p2);
    let d3 = query_sdf(p3);
    let d4 = query_sdf(p4);
    
    // Compute gradient from tetrahedral samples
    // This is more isotropic than axis-aligned sampling
    let grad = vec3<f32>(
        (d1 + d2 - d3 - d4) * 0.25,
        (d1 + d3 - d2 - d4) * 0.25,
        (d1 + d4 - d2 - d3) * 0.25
    );
    
    let gradient_length_sq = dot(grad, grad);
    
    // Handle edge cases where gradient is near zero (flat surfaces)
    let eps_sq = eps * eps * 0.0001;
    if (gradient_length_sq < eps_sq) {
        return vec3<f32>(0.0, 0.0, 1.0); // Default upward normal
    }
    
    return grad / sqrt(gradient_length_sq);
}

// Fast normal calculation for interactive operations (sculpting, raycasts)
fn calculate_normal_fast(world_pos: vec3<f32>) -> vec3<f32> {
    // PERFORMANCE OPTIMIZED: Simplified normal calculation for interactive operations
    let eps = get_effective_voxel_size() * 0.6; // Larger epsilon for fewer SDF queries
    
    // Simplified gradient calculation - fewer queries for speed
    let dx = query_sdf(world_pos + vec3<f32>(eps, 0.0, 0.0)) - query_sdf(world_pos - vec3<f32>(eps, 0.0, 0.0));
    let dy = query_sdf(world_pos + vec3<f32>(0.0, eps, 0.0)) - query_sdf(world_pos - vec3<f32>(0.0, eps, 0.0));
    let dz = query_sdf(world_pos + vec3<f32>(0.0, 0.0, eps)) - query_sdf(world_pos - vec3<f32>(0.0, 0.0, eps));
    
    return normalize(vec3<f32>(dx, dy, dz));
}

// uniforms from main raymarcher - using proper WGSL alignment
struct Uniforms {
    time: f32,
    resolution: vec2<f32>,
    cameraPosition: vec3<f32>,
    cameraMatrix: mat4x4<f32>,
    projectionMatrix: mat4x4<f32>,
    maxSteps: f32,
    maxDistance: f32,
    surfaceEpsilon: f32,
    // adaptive ray marching uniforms
    adaptiveEnabled: f32,
    adaptiveQualityMultiplier: f32,
    adaptiveMinStepMultiplier: f32,
    adaptiveMaxStepMultiplier: f32,
    adaptiveComplexityThreshold: f32,
    adaptiveDistanceScale: f32,
    showBrushPreview: f32,
    brushPosition: vec3<f32>,
    brushRadius: f32,
    // lens distortion parameters
    lensDistortionEnabled: f32,
    alwaysCalculateNormals: f32, // config for normal calculation strategy
    principalPoint: vec2<f32>,
    lensDistortion: vec4<f32>, // k1, k2, p1, p2
    // Add view matrix for matcap
    viewMatrix: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var matcap_texture: texture_2d<f32>;
@group(0) @binding(2) var texture_sampler: sampler;

// ggx/trowbridge-reitz distribution
fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h = max(dot(n, h), 0.0);
    let n_dot_h2 = n_dot_h * n_dot_h;

    let num = a2;
    let denom = (n_dot_h2 * (a2 - 1.0) + 1.0);
    let denom2 = denom * denom;

    return num / (3.14159265 * denom2);
}

// geometry function (smith's method)
fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;

    let num = n_dot_v;
    let denom = n_dot_v * (1.0 - k) + k;

    return num / denom;
}

// geometry function for smith's method
fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    let ggx2 = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx1 = geometry_schlick_ggx(n_dot_l, roughness);

    return ggx1 * ggx2;
}

// fresnel (schlick approximation)
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// pbr lighting with GGX specular (keeping for reference)
fn pbr_lighting(position: vec3<f32>, normal: vec3<f32>, view_dir: vec3<f32>, base_color: vec3<f32>) -> vec3<f32> {
    // n dot eye for headlight
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    var final_color = vec3<f32>(0.18, 0.18, 0.18) * n_dot_v;

    // add ggx specular
    let roughness = 0.4; // fixed roughness for pbr look
    let f0 = vec3<f32>(0.04); // fixed fresnel reflectance
    let h = normalize(view_dir + normal); // half vector
    let n_dot_h = max(dot(normal, h), 0.0);
    let d = distribution_ggx(normal, h, roughness);
    let g = geometry_smith(normal, view_dir, normal, roughness);
    let f = fresnel_schlick(n_dot_v, f0);
    let specular = (d * f * g) / max(4.0 * n_dot_v * n_dot_h, 0.001);

    // combine diffuse and specular
    let k_d = (1.0 - 0.5); // fixed diffuse factor
    final_color += k_d * base_color * n_dot_v; // diffuse component
    final_color += specular * (1.0 - 0.5); // specular component, fade out with distance
    final_color = clamp(final_color, vec3<f32>(0.0), vec3<f32>(1.0)); // ensure color is in valid range

    return final_color;
}

// scene SDF that uses VoxelHashMap
fn sceneSDF(p: vec3<f32>, time: f32) -> f32 {
    return query_sdf(p);
}


@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_triangle_vertex(vertex_index);
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
}

// apply lens distortion to normalized uv coordinates [0, 1]
fn applyLensDistortion(uv: vec2<f32>) -> vec2<f32> {
    // get distortion coefficients from uniforms
    let k1 = uniforms.lensDistortion.x; // radial distortion k1
    let k2 = uniforms.lensDistortion.y; // radial distortion k2
    let p1 = uniforms.lensDistortion.z; // tangential distortion p1
    let p2 = uniforms.lensDistortion.w; // tangential distortion p2

    // convert to coordinates centered at principal point
    let principalPoint = uniforms.principalPoint;
    let centeredUV = (uv - principalPoint) * 2.0; // scale to match typical lens model range

    // calculate radius squared
    let r2 = dot(centeredUV, centeredUV);
    let r4 = r2 * r2;
    let r6 = r4 * r2;

    // radial distortion (brown's model)
    let radialDistortion = 1.0 + k1 * r2 + k2 * r4;

    // tangential distortion
    let tangentialX = 2.0 * p1 * centeredUV.x * centeredUV.y + p2 * (r2 + 2.0 * centeredUV.x * centeredUV.x);
    let tangentialY = p1 * (r2 + 2.0 * centeredUV.y * centeredUV.y) + 2.0 * p2 * centeredUV.x * centeredUV.y;

    // apply distortion
    var distortedUV = centeredUV * radialDistortion + vec2<f32>(tangentialX, tangentialY);

    // convert back to UV space [0, 1]
    return distortedUV * 0.5 + principalPoint;
}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> FragmentOutput {
    var uv = vec2<f32>(
        (fragCoord.x / uniforms.resolution.x) * 2.0 - 1.0,
        1.0 - (fragCoord.y / uniforms.resolution.y) * 2.0
    );

    // EFFICIENCY: Only apply lens distortion computation if enabled
    if (uniforms.lensDistortionEnabled > 0.5) {
        // convert UV from [-1, 1] to [0, 1]
        let normalizedUV = vec2<f32>(
            fragCoord.x / uniforms.resolution.x,
            fragCoord.y / uniforms.resolution.y
        );
        // apply distortion
        let distortedUV = applyLensDistortion(normalizedUV);
        // convert back to [-1, 1] with proper y-flip
        uv = vec2<f32>(
            distortedUV.x * 2.0 - 1.0,
            1.0 - distortedUV.y * 2.0
        );
    }

    let aspectRatio = uniforms.resolution.x / uniforms.resolution.y;

    // use shared camera calculations
    let camBasis = get_camera_basis(uniforms.cameraMatrix);
    let tanHalfFov = get_fov_from_projection(uniforms.projectionMatrix);
    let ray_dir = generate_ray_direction(
        uv, aspectRatio, tanHalfFov,
        camBasis[0], camBasis[1], camBasis[2]
    );

    // Use full maxSteps for distance-dependent efficiency
    let adaptiveMaxSteps = i32(uniforms.maxSteps);

    var firstHit = trace_ray(
        uniforms.cameraPosition,
        ray_dir,
        adaptiveMaxSteps,
        uniforms.maxDistance,
        get_effective_voxel_size()
    );

    var hitPosition = vec3<f32>(0.0);
    var hitNormal = vec3<f32>(0.0, 0.0, 0.0);
    var hitDistance = uniforms.maxDistance;

    if (firstHit.hit) {
        hitPosition = firstHit.position;
        hitDistance = firstHit.distance;

        // Use fast normal calculation at distance (>10m) to reduce SDF queries
        // Use high-quality normal only for close-up details (<10m)
        if (hitDistance > 10.0) {
            hitNormal = calculate_normal_fast(hitPosition);
        } else {
            hitNormal = calculate_normal(hitPosition);
        }
    }
    
    // Apply matcap shading or background color
    var color = vec3<f32>(0.18, 0.18, 0.18); // background color

    if (false) {
        if (firstHit.hit) {
            color = pbr_lighting(
                hitPosition,
                hitNormal,
                -ray_dir,
                vec3<f32>(1.0)
            );
        }
    } else {
        // Transform normal to view space
        let view_space_normal = normalize((uniforms.viewMatrix * vec4<f32>(hitNormal, 0.0)).xyz);
        // Flip Y coordinate for correct matcap orientation
        let matcap_uv = vec2<f32>(view_space_normal.x * 0.5 + 0.5, 1.0 - (view_space_normal.y * 0.5 + 0.5));
        // Sample matcap texture
        let matcap_color = textureSample(matcap_texture, texture_sampler, matcap_uv).rgb;
        if (firstHit.hit) {
            color = matcap_color;
        }
    }


    var output: FragmentOutput;
    output.color = vec4<f32>(color, 1.0);
    return output;
}
`,_t=`
struct Uniforms {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    brush_position: vec3<f32>,
    brush_radius: f32,
    brush_normal: vec3<f32>,
    show_preview: f32,
    camera_position: vec3<f32>,
    _padding: f32
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) local_pos: vec2<f32>
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;

    if (uniforms.show_preview < 0.5) {
        output.position = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        return output;
    }

    // generate quad vertices
    let positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0)
    );

    let pos = positions[vertex_index];
    output.uv = pos * 0.5 + 0.5;
    output.local_pos = pos;

    // create basis vectors aligned to surface normal
    let normal = normalize(uniforms.brush_normal);

    // create tangent vector - avoid parallel to normal
    var tangent = vec3<f32>(1.0, 0.0, 0.0);
    if (abs(dot(normal, tangent)) > 0.9) {
        tangent = vec3<f32>(0.0, 0.0, 1.0);
    }
    tangent = normalize(tangent - normal * dot(tangent, normal));
    let bitangent = cross(normal, tangent);

    // scale and position the quad aligned to surface
    let scaled_radius = uniforms.brush_radius; // use full brush radius
    let world_pos = uniforms.brush_position +
                   tangent * pos.x * scaled_radius +
                   bitangent * pos.y * scaled_radius +
                   normal * 0.001; // very small offset from surface

    // transform through view and projection separately for debugging
    let view_pos = uniforms.view_matrix * vec4<f32>(world_pos, 1.0);
    output.position = uniforms.proj_matrix * view_pos;

    return output;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
    if (uniforms.show_preview < 0.5) {
        discard;
    }

    // distance from center
    let dist = length(input.local_pos);

    // draw circle outline at edge of preview
    let circle_thickness = 0.015; // much thinner
    let circle_radius = 0.98; // very close to edge
    let circle_dist = abs(dist - circle_radius);
    let circle = 1.0 - smoothstep(circle_thickness, circle_thickness * 0.5, circle_dist); // invert to make the ring

    // draw center dot - smaller
    let center_size = 0.04;
    let center_point = smoothstep(center_size, 0.0, dist);

    // draw crosshair - thinner
    let line_thickness = 0.01;
    let h_line = smoothstep(line_thickness, 0.0, abs(input.local_pos.y)) * step(abs(input.local_pos.x), 0.2);
    let v_line = smoothstep(line_thickness, 0.0, abs(input.local_pos.x)) * step(abs(input.local_pos.y), 0.2);
    let crosshair = max(h_line, v_line);

    // combine with full opacity for difference blending
    let alpha = max(max(circle, center_point), crosshair);

    // discard pixels outside circle
    if (dist > 1.0) {
        discard;
    }

    var output: FragmentOutput;
    // output color for difference blending - brighter colors work better
    // using a bright cyan that will invert nicely against most backgrounds
    output.color = vec4<f32>(0.3, 0.1, 0.2, alpha);
    return output;
}
`,vt=(r,e)=>{const t=r.createShaderModule({label:"Brush Preview Shader",code:_t}),i=r.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]}),s=r.createPipelineLayout({bindGroupLayouts:[i]});return{pipeline:r.createRenderPipeline({label:"Brush Preview Pipeline",layout:s,vertex:{module:t,entryPoint:"vs_main"},fragment:{module:t,entryPoint:"fs_main",targets:[{format:e,blend:{color:{srcFactor:"one-minus-dst",dstFactor:"one-minus-src",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list",cullMode:"none"}}),bindGroupLayout:i}};var ye=typeof Float32Array<"u"?Float32Array:Array;Math.hypot||(Math.hypot=function(){for(var r=0,e=arguments.length;e--;)r+=arguments[e]*arguments[e];return Math.sqrt(r)});function M(){var r=new ye(3);return ye!=Float32Array&&(r[0]=0,r[1]=0,r[2]=0),r}function Z(r){var e=new ye(3);return e[0]=r[0],e[1]=r[1],e[2]=r[2],e}function G(r,e,t){var i=new ye(3);return i[0]=r,i[1]=e,i[2]=t,i}function ce(r,e){return r[0]=e[0],r[1]=e[1],r[2]=e[2],r}function xe(r,e,t,i){return r[0]=e,r[1]=t,r[2]=i,r}function Ee(r,e,t){return r[0]=e[0]+t[0],r[1]=e[1]+t[1],r[2]=e[2]+t[2],r}function be(r,e,t){return r[0]=e[0]-t[0],r[1]=e[1]-t[1],r[2]=e[2]-t[2],r}function je(r,e,t){return r[0]=Math.min(e[0],t[0]),r[1]=Math.min(e[1],t[1]),r[2]=Math.min(e[2],t[2]),r}function Ke(r,e,t){return r[0]=Math.max(e[0],t[0]),r[1]=Math.max(e[1],t[1]),r[2]=Math.max(e[2],t[2]),r}function ne(r,e,t,i){return r[0]=e[0]+t[0]*i,r[1]=e[1]+t[1]*i,r[2]=e[2]+t[2]*i,r}function Je(r,e){var t=e[0]-r[0],i=e[1]-r[1],s=e[2]-r[2];return Math.hypot(t,i,s)}function oe(r,e){var t=e[0],i=e[1],s=e[2],a=t*t+i*i+s*s;return a>0&&(a=1/Math.sqrt(a)),r[0]=e[0]*a,r[1]=e[1]*a,r[2]=e[2]*a,r}function ve(r,e){return r[0]*e[0]+r[1]*e[1]+r[2]*e[2]}function de(r,e,t){var i=e[0],s=e[1],a=e[2],n=t[0],o=t[1],l=t[2];return r[0]=s*l-a*o,r[1]=a*n-i*l,r[2]=i*o-s*n,r}var ee=be;(function(){var r=M();return function(e,t,i,s,a,n){var o,l;for(t||(t=3),i||(i=0),s?l=Math.min(s*t+i,e.length):l=e.length,o=i;o<l;o+=t)r[0]=e[o],r[1]=e[o+1],r[2]=e[o+2],a(r,r,n),e[o]=r[0],e[o+1]=r[1],e[o+2]=r[2];return e}})();class yt{constructor(){this.listeners=new Map}on(e,t){if(typeof t!="function")throw new Error("Event handler must be a function");return this.listeners.has(e)||this.listeners.set(e,new Set),this.listeners.get(e).add(t),()=>this.off(e,t)}off(e,t){const i=this.listeners.get(e);i&&(i.delete(t),i.size===0&&this.listeners.delete(e))}emit(e,t=null){const i=this.listeners.get(e);if(!i)return 0;let s=0;for(const a of i)a(t),s++;return s}}const L=new yt,k={RENDER_START:"render:start",RENDER_COMPLETE:"render:complete",RESIZE:"viewport:resize",CAMERA_UPDATE:"camera:update",CAMERA_RESET:"camera:reset",SCULPT_START:"sculpt:start",SCULPT_END:"sculpt:end",BRUSH_CHANGE:"sculpt:brushChange",GPU_ERROR:"gpu:error"};class xt{constructor(){const e=d.camera;this.view={distance:e.defaultDistance,theta:e.defaultTheta,phi:e.defaultPhi,target:[...e.defaultTarget]},this.position=new Float32Array(3),this.viewMatrix=new Float32Array(16),this.projectionMatrix=new Float32Array(16),this.isDirty=!0}updateFromNavigation(e){if(this.view.theta+=e.x,this.view.phi+=e.y,this.view.distance+=e.zoom,e.panX!==0||e.panY!==0){const i=this.calculatePosition(),s=this.view.target,a=[0,1,0],n=M();be(n,s,i),oe(n,n);const o=M();de(o,n,a),oe(o,o);const l=M();de(l,o,n),this.view.target[0]+=o[0]*e.panX+l[0]*e.panY,this.view.target[1]+=o[1]*e.panX+l[1]*e.panY,this.view.target[2]+=o[2]*e.panX+l[2]*e.panY}const t=d.camera;this.view.phi=Math.max(t.minPhi,Math.min(t.maxPhi,this.view.phi)),this.view.distance=Math.max(t.minDistance,Math.min(t.maxDistance,this.view.distance)),this.isDirty=!0,L.emit(k.CAMERA_UPDATE,{position:this.calculatePosition(),target:[...this.view.target],distance:this.view.distance})}calculatePosition(){const e=this.view.distance,t=this.view.theta,i=this.view.phi;return[this.view.target[0]+e*Math.sin(i)*Math.cos(t),this.view.target[1]+e*Math.cos(i),this.view.target[2]+e*Math.sin(i)*Math.sin(t)]}createViewMatrix(e){const t=this.view.target,i=[0,1,0],s=M();be(s,t,e),oe(s,s);const a=M();de(a,s,i),oe(a,a);const n=M();return de(n,a,s),new Float32Array([a[0],n[0],-s[0],0,a[1],n[1],-s[1],0,a[2],n[2],-s[2],0,-ve(a,e),-ve(n,e),ve(s,e),1])}createProjectionMatrix(e,t){const i=d.camera;let s=e/t;if(i.aspectRatioMode==="fixed")s=i.fixedAspectRatio;else if(i.aspectRatioMode==="constrain"){const l=e/t;l>i.fixedAspectRatio?s=i.fixedAspectRatio:s=l}const a=i.nearPlane,n=i.farPlane;let o;if(i.projectionType==="perspective"){let l;if(i.sensorWidth&&i.focalLength){const v=i.sensorHeight;l=2*Math.atan(v/(2*i.focalLength))}else l=i.fov*Math.PI/180;const c=1/Math.tan(l/2),p=1/(a-n),u=(i.principalPointX-.5)*2,g=(i.principalPointY-.5)*2;o=new Float32Array([c/s,0,0,0,0,c,0,0,u,g,(a+n)*p,-1,0,0,a*n*p*2,0])}else if(i.projectionType==="orthographic"){const l=i.orthoSize,c=l*s,p=1/(a-n);o=new Float32Array([1/c,0,0,0,0,1/l,0,0,0,0,2*p,0,0,0,(a+n)*p,1])}else throw new Error(`Unknown projection type: ${i.projectionType}`);return o}update(e,t){if(!this.isDirty&&this.lastCanvasWidth===e&&this.lastCanvasHeight===t)return!1;const i=this.calculatePosition();return this.position[0]=i[0],this.position[1]=i[1],this.position[2]=i[2],this.viewMatrix=this.createViewMatrix(i),(this.isDirty||this.lastCanvasWidth!==e||this.lastCanvasHeight!==t)&&(this.projectionMatrix=this.createProjectionMatrix(e,t),this.lastCanvasWidth=e,this.lastCanvasHeight=t),this.isDirty=!1,!0}getCameraRight(){const e=this.view.theta;return[Math.sin(e),0,-Math.cos(e)]}reset(){const e=d.camera;this.view.theta=e.defaultTheta,this.view.phi=e.defaultPhi,this.view.distance=e.defaultDistance,this.view.target=[...e.defaultTarget],this.isDirty=!0,L.emit(k.CAMERA_RESET,{position:this.calculatePosition(),target:[...this.view.target],distance:this.view.distance})}}class Ze{constructor(){this.adapter=null,this.device=null,this.context=null,this.presentationFormat=null,this.workgroupLimits=null,this.deviceLimits=null,this.isInitialized=!1,this.contextLostHandlers=[],this.contextRestoredHandlers=[]}async initialize(e,t=[],i={}){if(this.isInitialized)return;if(!navigator.gpu){const a="WebGPU is not supported in this browser/environment";throw new Error(a)}await this.requestAdapter(),await this.requestDevice(t,i),await this.configureContext(e);const{initializeBindGroupFactory:s}=await ut(async()=>{const{initializeBindGroupFactory:a}=await Promise.resolve().then(()=>ri);return{initializeBindGroupFactory:a}},void 0,import.meta.url);s(this.device),this.isInitialized=!0}async requestAdapter(){if(this.adapter=await navigator.gpu.requestAdapter(),!this.adapter){const t=`No WebGPU adapter found. This might be due to:
- WebGPU not enabled in browser
- No compatible GPU
- Driver issues`;throw new Error(t)}const e=this.adapter.limits;this.workgroupLimits={maxX:e.maxComputeWorkgroupSizeX,maxY:e.maxComputeWorkgroupSizeY,maxZ:e.maxComputeWorkgroupSizeZ,maxInvocations:e.maxComputeInvocationsPerWorkgroup}}async requestDevice(e=[],t={}){const i=this.adapter.limits,s={requiredFeatures:e,requiredLimits:{maxBufferSize:Math.min(t.maxBufferSize||i.maxBufferSize,i.maxBufferSize),maxStorageBufferBindingSize:Math.min(t.maxStorageBufferBindingSize||i.maxStorageBufferBindingSize,i.maxStorageBufferBindingSize),...t}};if(t.maxStorageBuffersPerShaderStage){const a=t.maxStorageBuffersPerShaderStage,n=i.maxStorageBuffersPerShaderStage;a<=n&&(s.requiredLimits.maxStorageBuffersPerShaderStage=a)}i.maxColorAttachmentBytesPerSample&&i.maxColorAttachmentBytesPerSample>=64&&(s.requiredLimits.maxColorAttachmentBytesPerSample=64),this.device=await this.adapter.requestDevice(s),this.deviceLimits=this.device.limits,this.device.addEventListener("uncapturederror",a=>{L.emit(k.GPU_ERROR,{error:a.error,message:a.error.message,timestamp:Date.now()})}),this.device.lost.then(a=>{this.handleContextLost(a)})}async configureContext(e){if(this.context=e.getContext("webgpu"),!this.context)throw new Error("Failed to get WebGPU context from canvas");this.presentationFormat=navigator.gpu.getPreferredCanvasFormat(),this.context.configure({device:this.device,format:this.presentationFormat,alphaMode:"premultiplied"})}handleContextLost(e){this.isInitialized=!1;for(const t of this.contextLostHandlers)t(e)}onContextLost(e){this.contextLostHandlers.push(e)}offContextLost(e){const t=this.contextLostHandlers.indexOf(e);t!==-1&&this.contextLostHandlers.splice(t,1)}getCurrentTexture(){if(!this.context)throw new Error("Context not initialized");return this.context.getCurrentTexture()}async createShaderModule(e){const t=this.device.createShaderModule(e),i=await t.getCompilationInfo();if(i.messages.length>0){let s=!1;if(i.messages.forEach(a=>{if(a.type==="error"&&(s=!0),a.lineNum&&e.code){const n=e.code.split(`
`),o=Math.max(0,a.lineNum-3),l=Math.min(n.length,a.lineNum+2);for(let c=o;c<l;c++)a.lineNum-1}}),s)throw new Error("Shader compilation failed with errors")}return t}hasFeature(e){return this.device&&this.device.features.has(e)}getLimit(e){return this.deviceLimits?this.deviceLimits[e]:null}destroy(){this.device&&(this.device.destroy(),this.device=null),this.context&&(this.context.unconfigure(),this.context=null),this.adapter=null,this.isInitialized=!1,this.contextLostHandlers=[],this.contextRestoredHandlers=[]}}class bt{constructor(e){this.canvas=e,this.enabled=!0,this.mouse={x:0,y:0,lastX:0,lastY:0,deltaX:0,deltaY:0,button:-1,buttons:0,wheel:0},this.activeDragAction=null,this.keys=new Set,this.modifiers={shift:!1,ctrl:!1,alt:!1,meta:!1},this.touches=new Map,this.lastTouchDistance=0,this.pointerLocked=!1,this.actionHandlers=new Map,this.gestureState={pinching:!1,panning:!1,rotating:!1},this.setupEventListeners(),this.setupDefaultMappings()}setupEventListeners(){this.canvas.addEventListener("mousedown",this.handleMouseDown.bind(this)),window.addEventListener("mousemove",this.handleMouseMove.bind(this)),window.addEventListener("mouseup",this.handleMouseUp.bind(this)),this.canvas.addEventListener("wheel",this.handleWheel.bind(this),{passive:!1}),this.canvas.addEventListener("contextmenu",e=>e.preventDefault()),this.canvas.addEventListener("touchstart",this.handleTouchStart.bind(this),{passive:!1}),this.canvas.addEventListener("touchmove",this.handleTouchMove.bind(this),{passive:!1}),this.canvas.addEventListener("touchend",this.handleTouchEnd.bind(this)),window.addEventListener("keydown",this.handleKeyDown.bind(this)),window.addEventListener("keyup",this.handleKeyUp.bind(this)),document.addEventListener("pointerlockchange",this.handlePointerLockChange.bind(this)),window.addEventListener("blur",this.handleWindowBlur.bind(this))}setupDefaultMappings(){this.mapMouseDrag(0,"camera:rotate"),this.mapMouseDrag(1,"camera:pan"),this.mapMouseDrag(2,"camera:pan"),this.mapWheel("camera:zoom"),this.mapKey(" ","animation:toggle"),this.mapKey("r","camera:reset")}onAction(e,t,i={}){this.actionHandlers.has(e)||this.actionHandlers.set(e,[]);const s=this.actionHandlers.get(e);s.push({handler:t,options:i}),s.sort((a,n)=>(n.options.priority||0)-(a.options.priority||0))}offAction(e,t){const i=this.actionHandlers.get(e);if(!i)return;const s=i.findIndex(a=>a.handler===t);s!==-1&&i.splice(s,1)}triggerAction(e,t={}){if(!this.enabled)throw new Error(`Input system is disabled, cannot trigger action: ${e}`);const i=this.actionHandlers.get(e);if(!i||i.length===0)throw new Error(`No handlers registered for action: ${e}`);const s={...t,mouse:{...this.mouse},keys:new Set(this.keys),modifiers:{...this.modifiers},timestamp:performance.now(),activeDragAction:this.activeDragAction};let a=!1;for(const{handler:n,options:o}of i)if(n(s)===!1){a=!0;break}return a?!1:void 0}handleMouseDown(e){this.mouse.button=e.button,this.mouse.buttons=e.buttons,this.updateMousePosition(e);const t=this.getMouseAction("drag",e.button),i=this.getMouseAction("down",e.button);let s=!1;i&&(s=this.triggerAction(i,{event:e,activeDragAction:t})===!1),s?this.activeDragAction=null:this.activeDragAction=t,e.preventDefault()}handleMouseMove(e){const t=this.mouse.x,i=this.mouse.y;this.updateMousePosition(e),this.mouse.deltaX=this.mouse.x-t,this.mouse.deltaY=this.mouse.y-i,this.mouse.button!==-1&&this.activeDragAction&&this.triggerAction(this.activeDragAction,{event:e});const s=this.getMouseAction("move");s&&this.triggerAction(s,{event:e})}handleMouseUp(e){const t=this.getMouseAction("up",e.button);t&&this.triggerAction(t,{event:e}),this.mouse.button=-1,this.mouse.buttons=e.buttons,this.activeDragAction=null}handleWheel(e){e.preventDefault(),this.mouse.wheel=e.deltaY;const t=this.getWheelAction();t&&this.triggerAction(t,{event:e,delta:e.deltaY})}handleTouchStart(e){e.preventDefault();for(const t of e.changedTouches)this.touches.set(t.identifier,{x:t.clientX,y:t.clientY,startX:t.clientX,startY:t.clientY});this.updateGestureState()}handleTouchMove(e){e.preventDefault();for(const t of e.changedTouches){const i=this.touches.get(t.identifier);i&&(i.x=t.clientX,i.y=t.clientY)}this.updateGestureState(),this.handleGestures()}handleTouchEnd(e){for(const t of e.changedTouches)this.touches.delete(t.identifier);this.updateGestureState()}handleKeyDown(e){this.keys.add(e.code),this.updateModifiers(e);const t=this.getKeyAction(e.code);t&&(this.triggerAction(t,{event:e,key:e.code}),e.preventDefault())}handleKeyUp(e){this.keys.delete(e.code),this.updateModifiers(e)}updateMousePosition(e){const t=this.canvas.getBoundingClientRect();this.mouse.lastX=this.mouse.x,this.mouse.lastY=this.mouse.y,this.mouse.x=e.clientX-t.left,this.mouse.y=e.clientY-t.top}updateModifiers(e){this.modifiers.shift=e.shiftKey,this.modifiers.ctrl=e.ctrlKey||e.metaKey,this.modifiers.alt=e.altKey,this.modifiers.meta=e.metaKey}updateGestureState(){const e=this.touches.size;this.gestureState.pinching=e===2,this.gestureState.panning=e===2||e===3,this.gestureState.rotating=e===2}handleGestures(){if(this.gestureState.pinching&&this.touches.size===2){const e=Array.from(this.touches.values()),t=Math.hypot(e[1].x-e[0].x,e[1].y-e[0].y);if(this.lastTouchDistance>0){const i=t/this.lastTouchDistance;this.triggerAction("camera:zoom",{delta:(1-i)*100,gesture:"pinch"})}this.lastTouchDistance=t}}handlePointerLockChange(){this.pointerLocked=document.pointerLockElement===this.canvas}handleWindowBlur(){this.mouse.button=-1,this.mouse.buttons=0,this.keys.clear(),this.touches.clear(),this.updateModifiers({shiftKey:!1,ctrlKey:!1,altKey:!1,metaKey:!1})}mapMouseDown(e,t){this.mouseDownMap=this.mouseDownMap||new Map,this.mouseDownMap.set(e,t)}mapMouseUp(e,t){this.mouseUpMap=this.mouseUpMap||new Map,this.mouseUpMap.set(e,t)}mapMouseDrag(e,t){this.mouseDragMap=this.mouseDragMap||new Map,this.mouseDragMap.set(e,t)}mapMouseMove(e){this.mouseMoveAction=e}mapWheel(e){this.wheelAction=e}mapKey(e,t){this.keyMap=this.keyMap||new Map,this.keyMap.set(e,t)}getMouseAction(e,t){var i,s,a;switch(e){case"down":return(i=this.mouseDownMap)==null?void 0:i.get(t);case"up":return(s=this.mouseUpMap)==null?void 0:s.get(t);case"drag":return(a=this.mouseDragMap)==null?void 0:a.get(t);case"move":return this.mouseMoveAction}}getWheelAction(){return this.wheelAction}getKeyAction(e){var i,s;const t=e.replace("Key","").toLowerCase();return((i=this.keyMap)==null?void 0:i.get(t))||((s=this.keyMap)==null?void 0:s.get(e))}getNormalizedMousePosition(){const e=window.devicePixelRatio,t=this.mouse.x*e,i=this.mouse.y*e;return{x:t/this.canvas.width*2-1,y:-(i/this.canvas.height*2-1)}}isKeyPressed(e){return this.keys.has(e)||this.keys.has("Key"+e.toUpperCase())}enable(){this.enabled=!0}disable(){this.enabled=!1}destroy(){this.canvas.removeEventListener("mousedown",this.handleMouseDown),window.removeEventListener("mousemove",this.handleMouseMove),window.removeEventListener("mouseup",this.handleMouseUp),this.canvas.removeEventListener("wheel",this.handleWheel),this.canvas.removeEventListener("contextmenu",e=>e.preventDefault()),this.canvas.removeEventListener("touchstart",this.handleTouchStart),this.canvas.removeEventListener("touchmove",this.handleTouchMove),this.canvas.removeEventListener("touchend",this.handleTouchEnd),window.removeEventListener("keydown",this.handleKeyDown),window.removeEventListener("keyup",this.handleKeyUp),document.removeEventListener("pointerlockchange",this.handlePointerLockChange),window.removeEventListener("blur",this.handleWindowBlur),this.actionHandlers.clear(),this.keys.clear(),this.touches.clear()}}const h={FLOAT:"float",VEC2:"vec2",VEC3:"vec3",VEC4:"vec4",MAT3:"mat3",MAT4:"mat4",INT:"int",UINT:"uint",BOOL:"bool"},te={PER_FRAME:"per_frame",ON_CHANGE:"on_change"};function wt(r){return{[h.FLOAT]:4,[h.VEC2]:8,[h.VEC3]:12,[h.VEC4]:16,[h.MAT3]:48,[h.MAT4]:64,[h.INT]:4,[h.UINT]:4,[h.BOOL]:4}[r]||0}function St(r){return{[h.FLOAT]:4,[h.VEC2]:8,[h.VEC3]:16,[h.VEC4]:16,[h.MAT3]:16,[h.MAT4]:16,[h.INT]:4,[h.UINT]:4,[h.BOOL]:4}[r]||4}function Ne(r,e){return Math.ceil(r/e)*e}class Pt{constructor(e,t,i,s=te.ON_CHANGE){this.device=e,this.name=t,this.updateFrequency=s,this.isDirty=!0,this.uniforms=new Map,this.layout=[];let a=0;for(const n of i){const o=St(n.type),l=Ne(a,o),c=wt(n.type),p={name:n.name,type:n.type,offset:l,size:c,defaultValue:n.defaultValue||this.getDefaultValue(n.type),validator:n.validator};this.uniforms.set(n.name,p),this.layout.push(p),a=l+c}this.bufferSize=Ne(a,16),this.gpuBuffer=e.createBuffer({label:`UniformBuffer_${t}`,size:this.bufferSize,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.cpuData=new ArrayBuffer(this.bufferSize),this.dataViews=this.createDataViews(),this.initializeDefaults()}createDataViews(){const e={};for(const[t,i]of this.uniforms)switch(i.type){case h.FLOAT:case h.INT:e[t]=new Float32Array(this.cpuData,i.offset,1);break;case h.UINT:e[t]=new Uint32Array(this.cpuData,i.offset,1);break;case h.BOOL:e[t]=new Uint32Array(this.cpuData,i.offset,1);break;case h.VEC2:e[t]=new Float32Array(this.cpuData,i.offset,2);break;case h.VEC3:e[t]=new Float32Array(this.cpuData,i.offset,3);break;case h.VEC4:e[t]=new Float32Array(this.cpuData,i.offset,4);break;case h.MAT3:e[t]=new Float32Array(this.cpuData,i.offset,12);break;case h.MAT4:e[t]=new Float32Array(this.cpuData,i.offset,16);break}return e}getDefaultValue(e){switch(e){case h.FLOAT:case h.INT:case h.UINT:return 0;case h.BOOL:return!1;case h.VEC2:return[0,0];case h.VEC3:return[0,0,0];case h.VEC4:return[0,0,0,0];case h.MAT3:return[1,0,0,0,0,1,0,0,0,0,1,0];case h.MAT4:return[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];default:return null}}initializeDefaults(){for(const[e,t]of this.uniforms)this.setValue(e,t.defaultValue);this.isDirty=!0}setValue(e,t){const i=this.uniforms.get(e);if(!i||i.validator&&!i.validator(t))return;const s=this.dataViews[e];switch(i.type){case h.BOOL:s[0]=t?1:0;break;case h.FLOAT:case h.INT:case h.UINT:s[0]=t;break;case h.VEC2:case h.VEC3:case h.VEC4:s.set(t);break;case h.MAT3:for(let a=0;a<3;a++){for(let n=0;n<3;n++)s[a*4+n]=t[a*3+n];s[a*4+3]=0}break;case h.MAT4:s.set(t);break}this.isDirty=!0}getValue(e){const t=this.uniforms.get(e);if(!t)return null;const i=this.dataViews[e];switch(t.type){case h.BOOL:return i[0]!==0;case h.FLOAT:case h.INT:case h.UINT:return i[0];case h.VEC2:return[i[0],i[1]];case h.VEC3:return[i[0],i[1],i[2]];case h.VEC4:return[i[0],i[1],i[2],i[3]];case h.MAT3:const s=[];for(let a=0;a<3;a++)for(let n=0;n<3;n++)s.push(i[a*4+n]);return s;case h.MAT4:return Array.from(i)}}upload(e){return this.isDirty?(e.writeBuffer(this.gpuBuffer,0,this.cpuData),this.isDirty=!1,!0):!1}forceUpload(e){e.writeBuffer(this.gpuBuffer,0,this.cpuData),this.isDirty=!1}markDirty(){this.isDirty=!0}getGPUBuffer(){return this.gpuBuffer}destroy(){this.gpuBuffer&&(this.gpuBuffer.destroy(),this.gpuBuffer=null)}}class Bt{constructor(e){this.device=e,this.buffers=new Map,this.updateQueue=new Set,this.frameUpdateBuffers=new Set}registerBuffer(e,t,i=te.ON_CHANGE){if(this.buffers.has(e))return this.buffers.get(e);const s=new Pt(this.device,e,t,i);return this.buffers.set(e,s),i===te.PER_FRAME&&this.frameUpdateBuffers.add(e),s}getBuffer(e){return this.buffers.get(e)}setValue(e,t,i){const s=this.buffers.get(e);s&&(s.setValue(t,i),s.updateFrequency!==te.PER_FRAME&&this.updateQueue.add(e))}getValue(e,t){const i=this.buffers.get(e);return i?i.getValue(t):null}batchUpdate(e,t){const i=this.buffers.get(e);if(i){for(const[s,a]of Object.entries(t))i.setValue(s,a);i.updateFrequency!==te.PER_FRAME&&this.updateQueue.add(e)}}updateBuffers(e){const t=[];for(const i of this.frameUpdateBuffers){const s=this.buffers.get(i);s&&s.upload(e)&&t.push(i)}for(const i of this.updateQueue){const s=this.buffers.get(i);s&&s.upload(e)&&t.push(i)}return this.updateQueue.clear(),t}forceUpdateAll(e){for(const[t,i]of this.buffers)i.forceUpload(e)}createBindGroupEntry(e,t){const i=this.buffers.get(e);return i?{binding:t,resource:{buffer:i.getGPUBuffer()}}:null}getBufferMetadata(e){const t=this.buffers.get(e);return t?{name:t.name,size:t.bufferSize,uniforms:Array.from(t.uniforms.values()),updateFrequency:t.updateFrequency}:null}generateWGSLStruct(e,t){const i=this.buffers.get(e);if(!i)return"";let s=`struct ${t} {
`;for(const a of i.layout){const n=this.getWGSLType(a.type);s+=`    ${a.name}: ${n},
`}return s+=`};
`,s}getWGSLType(e){return{[h.FLOAT]:"f32",[h.VEC2]:"vec2<f32>",[h.VEC3]:"vec3<f32>",[h.VEC4]:"vec4<f32>",[h.MAT3]:"mat3x3<f32>",[h.MAT4]:"mat4x4<f32>",[h.INT]:"i32",[h.UINT]:"u32",[h.BOOL]:"u32"}[e]||"f32"}getMemoryStats(){let e=0;const t=[];for(const[i,s]of this.buffers)e+=s.bufferSize,t.push({name:i,size:s.bufferSize,uniformCount:s.uniforms.size,updateFrequency:s.updateFrequency});return{totalSize:e,bufferCount:this.buffers.size,buffers:t}}destroy(){for(const e of this.buffers.values())e.destroy();this.buffers.clear(),this.updateQueue.clear(),this.frameUpdateBuffers.clear()}}function R(r,e,t=null,i=null){return{name:r,type:e,defaultValue:t,validator:i}}const Mt={positiveFloat:r=>typeof r=="number"&&r>0};function Et(r){return[R("time",h.FLOAT,0),R("resolution",h.VEC2,[r.width,r.height]),R("cameraPosition",h.VEC3,[0,0,0]),R("cameraMatrix",h.MAT4,new Float32Array(16)),R("projectionMatrix",h.MAT4,new Float32Array(16)),R("maxSteps",h.FLOAT,d.raymarching.maxSteps),R("maxDistance",h.FLOAT,d.camera.farPlane),R("surfaceEpsilon",h.FLOAT,d.raymarching.surfaceEpsilon),R("adaptiveEnabled",h.FLOAT,d.raymarching.adaptive.enabled?1:0),R("adaptiveQualityMultiplier",h.FLOAT,d.raymarching.adaptive.qualityMultiplier),R("adaptiveMinStepMultiplier",h.FLOAT,d.raymarching.adaptive.minStepMultiplier),R("adaptiveMaxStepMultiplier",h.FLOAT,d.raymarching.adaptive.maxStepMultiplier),R("adaptiveComplexityThreshold",h.FLOAT,d.raymarching.adaptive.complexityThreshold),R("adaptiveDistanceScale",h.FLOAT,d.raymarching.adaptive.distanceScale),R("showBrushPreview",h.FLOAT,0),R("brushPosition",h.VEC3,[0,0,0]),R("brushRadius",h.FLOAT,.5,Mt.positiveFloat),R("lensDistortionEnabled",h.FLOAT,0),R("alwaysCalculateNormals",h.FLOAT,0),R("principalPoint",h.VEC2,[.5,.5]),R("lensDistortion",h.VEC4,[0,0,0,0]),R("viewMatrix",h.MAT4,new Float32Array(16))]}function At(r){var v,w;const e={};(((v=r.lastResolution)==null?void 0:v[0])!==r.canvas.width||((w=r.lastResolution)==null?void 0:w[1])!==r.canvas.height)&&(e.resolution=[r.canvas.width,r.canvas.height],r.lastResolution=[r.canvas.width,r.canvas.height]);const t=r.navigation.x!==0||r.navigation.y!==0||r.navigation.zoom!==0||r.navigation.panX!==0||r.navigation.panY!==0;(t||r.camera.isDirty)&&(r.camera.update(r.canvas.width,r.canvas.height),e.cameraMatrix=r.camera.viewMatrix,e.viewMatrix=r.camera.viewMatrix,e.projectionMatrix=r.camera.projectionMatrix,e.cameraPosition=Array.from(r.camera.position));const i=r.showBrushPreview?1:0;r.lastBrushPreviewState!==i&&(e.showBrushPreview=i,r.lastBrushPreviewState=i),r.showBrushPreview&&(e.brushPosition=r.brushPreviewPosition,e.brushRadius=r.brushPreviewRadius);const s=0;r.lastNormalStrategy!==s&&(e.alwaysCalculateNormals=s,r.lastNormalStrategy=s);const n=r.settings.lensDistortionEnabled?1:0;r.lastLensDistortionState!==n&&(e.lensDistortionEnabled=n,r.lastLensDistortionState=n);const o=d.raymarching,l=`${o.maxSteps}|${o.surfaceEpsilon}`;r.lastRayMarchingKey!==l&&(e.maxSteps=o.maxSteps,e.surfaceEpsilon=o.surfaceEpsilon,r.lastRayMarchingKey=l);const c=o.adaptive,p=`${c.enabled}|${c.qualityMultiplier}|${c.minStepMultiplier}|${c.maxStepMultiplier}|${c.complexityThreshold}|${c.distanceScale}`;r.lastAdaptiveKey!==p&&(e.adaptiveEnabled=c.enabled?1:0,e.adaptiveQualityMultiplier=c.qualityMultiplier,e.adaptiveMinStepMultiplier=c.minStepMultiplier,e.adaptiveMaxStepMultiplier=c.maxStepMultiplier,e.adaptiveComplexityThreshold=c.complexityThreshold,e.adaptiveDistanceScale=c.distanceScale,r.lastAdaptiveKey=p),r.lastMaxDistance!==d.camera.farPlane&&(e.maxDistance=d.camera.farPlane,r.lastMaxDistance=d.camera.farPlane);const u=d.camera,g=`${u.principalPointX}|${u.principalPointY}|${u.radialK1}|${u.radialK2}|${u.tangentialP1}|${u.tangentialP2}`;return r.lastLensKey!==g&&(e.principalPoint=[u.principalPointX,u.principalPointY],e.lensDistortion=[u.radialK1,u.radialK2,u.tangentialP1,u.tangentialP2],r.lastLensKey=g),{updates:e,hasNavigation:t}}/**
 * lil-gui
 * https://lil-gui.georgealways.com
 * @version 0.20.0
 * @author George Michael Brower
 * @license MIT
 */class Y{constructor(e,t,i,s,a="div"){this.parent=e,this.object=t,this.property=i,this._disabled=!1,this._hidden=!1,this.initialValue=this.getValue(),this.domElement=document.createElement(a),this.domElement.classList.add("controller"),this.domElement.classList.add(s),this.$name=document.createElement("div"),this.$name.classList.add("name"),Y.nextNameID=Y.nextNameID||0,this.$name.id=`lil-gui-name-${++Y.nextNameID}`,this.$widget=document.createElement("div"),this.$widget.classList.add("widget"),this.$disable=this.$widget,this.domElement.appendChild(this.$name),this.domElement.appendChild(this.$widget),this.domElement.addEventListener("keydown",n=>n.stopPropagation()),this.domElement.addEventListener("keyup",n=>n.stopPropagation()),this.parent.children.push(this),this.parent.controllers.push(this),this.parent.$children.appendChild(this.domElement),this._listenCallback=this._listenCallback.bind(this),this.name(i)}name(e){return this._name=e,this.$name.textContent=e,this}onChange(e){return this._onChange=e,this}_callOnChange(){this.parent._callOnChange(this),this._onChange!==void 0&&this._onChange.call(this,this.getValue()),this._changed=!0}onFinishChange(e){return this._onFinishChange=e,this}_callOnFinishChange(){this._changed&&(this.parent._callOnFinishChange(this),this._onFinishChange!==void 0&&this._onFinishChange.call(this,this.getValue())),this._changed=!1}reset(){return this.setValue(this.initialValue),this._callOnFinishChange(),this}enable(e=!0){return this.disable(!e)}disable(e=!0){return e===this._disabled?this:(this._disabled=e,this.domElement.classList.toggle("disabled",e),this.$disable.toggleAttribute("disabled",e),this)}show(e=!0){return this._hidden=!e,this.domElement.style.display=this._hidden?"none":"",this}hide(){return this.show(!1)}options(e){const t=this.parent.add(this.object,this.property,e);return t.name(this._name),this.destroy(),t}min(e){return this}max(e){return this}step(e){return this}decimals(e){return this}listen(e=!0){return this._listening=e,this._listenCallbackID!==void 0&&(cancelAnimationFrame(this._listenCallbackID),this._listenCallbackID=void 0),this._listening&&this._listenCallback(),this}_listenCallback(){this._listenCallbackID=requestAnimationFrame(this._listenCallback);const e=this.save();e!==this._listenPrevValue&&this.updateDisplay(),this._listenPrevValue=e}getValue(){return this.object[this.property]}setValue(e){return this.getValue()!==e&&(this.object[this.property]=e,this._callOnChange(),this.updateDisplay()),this}updateDisplay(){return this}load(e){return this.setValue(e),this._callOnFinishChange(),this}save(){return this.getValue()}destroy(){this.listen(!1),this.parent.children.splice(this.parent.children.indexOf(this),1),this.parent.controllers.splice(this.parent.controllers.indexOf(this),1),this.parent.$children.removeChild(this.domElement)}}class Ct extends Y{constructor(e,t,i){super(e,t,i,"boolean","label"),this.$input=document.createElement("input"),this.$input.setAttribute("type","checkbox"),this.$input.setAttribute("aria-labelledby",this.$name.id),this.$widget.appendChild(this.$input),this.$input.addEventListener("change",()=>{this.setValue(this.$input.checked),this._callOnFinishChange()}),this.$disable=this.$input,this.updateDisplay()}updateDisplay(){return this.$input.checked=this.getValue(),this}}function Fe(r){let e,t;return(e=r.match(/(#|0x)?([a-f0-9]{6})/i))?t=e[2]:(e=r.match(/rgb\(\s*(\d*)\s*,\s*(\d*)\s*,\s*(\d*)\s*\)/))?t=parseInt(e[1]).toString(16).padStart(2,0)+parseInt(e[2]).toString(16).padStart(2,0)+parseInt(e[3]).toString(16).padStart(2,0):(e=r.match(/^#?([a-f0-9])([a-f0-9])([a-f0-9])$/i))&&(t=e[1]+e[1]+e[2]+e[2]+e[3]+e[3]),t?"#"+t:!1}const Tt={isPrimitive:!0,match:r=>typeof r=="string",fromHexString:Fe,toHexString:Fe},ue={isPrimitive:!0,match:r=>typeof r=="number",fromHexString:r=>parseInt(r.substring(1),16),toHexString:r=>"#"+r.toString(16).padStart(6,0)},zt={isPrimitive:!1,match:r=>Array.isArray(r),fromHexString(r,e,t=1){const i=ue.fromHexString(r);e[0]=(i>>16&255)/255*t,e[1]=(i>>8&255)/255*t,e[2]=(i&255)/255*t},toHexString([r,e,t],i=1){i=255/i;const s=r*i<<16^e*i<<8^t*i<<0;return ue.toHexString(s)}},Ft={isPrimitive:!1,match:r=>Object(r)===r,fromHexString(r,e,t=1){const i=ue.fromHexString(r);e.r=(i>>16&255)/255*t,e.g=(i>>8&255)/255*t,e.b=(i&255)/255*t},toHexString({r,g:e,b:t},i=1){i=255/i;const s=r*i<<16^e*i<<8^t*i<<0;return ue.toHexString(s)}},Rt=[Tt,ue,zt,Ft];function Gt(r){return Rt.find(e=>e.match(r))}class Dt extends Y{constructor(e,t,i,s){super(e,t,i,"color"),this.$input=document.createElement("input"),this.$input.setAttribute("type","color"),this.$input.setAttribute("tabindex",-1),this.$input.setAttribute("aria-labelledby",this.$name.id),this.$text=document.createElement("input"),this.$text.setAttribute("type","text"),this.$text.setAttribute("spellcheck","false"),this.$text.setAttribute("aria-labelledby",this.$name.id),this.$display=document.createElement("div"),this.$display.classList.add("display"),this.$display.appendChild(this.$input),this.$widget.appendChild(this.$display),this.$widget.appendChild(this.$text),this._format=Gt(this.initialValue),this._rgbScale=s,this._initialValueHexString=this.save(),this._textFocused=!1,this.$input.addEventListener("input",()=>{this._setValueFromHexString(this.$input.value)}),this.$input.addEventListener("blur",()=>{this._callOnFinishChange()}),this.$text.addEventListener("input",()=>{const a=Fe(this.$text.value);a&&this._setValueFromHexString(a)}),this.$text.addEventListener("focus",()=>{this._textFocused=!0,this.$text.select()}),this.$text.addEventListener("blur",()=>{this._textFocused=!1,this.updateDisplay(),this._callOnFinishChange()}),this.$disable=this.$text,this.updateDisplay()}reset(){return this._setValueFromHexString(this._initialValueHexString),this}_setValueFromHexString(e){if(this._format.isPrimitive){const t=this._format.fromHexString(e);this.setValue(t)}else this._format.fromHexString(e,this.getValue(),this._rgbScale),this._callOnChange(),this.updateDisplay()}save(){return this._format.toHexString(this.getValue(),this._rgbScale)}load(e){return this._setValueFromHexString(e),this._callOnFinishChange(),this}updateDisplay(){return this.$input.value=this._format.toHexString(this.getValue(),this._rgbScale),this._textFocused||(this.$text.value=this.$input.value.substring(1)),this.$display.style.backgroundColor=this.$input.value,this}}class Ae extends Y{constructor(e,t,i){super(e,t,i,"function"),this.$button=document.createElement("button"),this.$button.appendChild(this.$name),this.$widget.appendChild(this.$button),this.$button.addEventListener("click",s=>{s.preventDefault(),this.getValue().call(this.object),this._callOnChange()}),this.$button.addEventListener("touchstart",()=>{},{passive:!0}),this.$disable=this.$button}}class Ut extends Y{constructor(e,t,i,s,a,n){super(e,t,i,"number"),this._initInput(),this.min(s),this.max(a);const o=n!==void 0;this.step(o?n:this._getImplicitStep(),o),this.updateDisplay()}decimals(e){return this._decimals=e,this.updateDisplay(),this}min(e){return this._min=e,this._onUpdateMinMax(),this}max(e){return this._max=e,this._onUpdateMinMax(),this}step(e,t=!0){return this._step=e,this._stepExplicit=t,this}updateDisplay(){const e=this.getValue();if(this._hasSlider){let t=(e-this._min)/(this._max-this._min);t=Math.max(0,Math.min(t,1)),this.$fill.style.width=t*100+"%"}return this._inputFocused||(this.$input.value=this._decimals===void 0?e:e.toFixed(this._decimals)),this}_initInput(){this.$input=document.createElement("input"),this.$input.setAttribute("type","text"),this.$input.setAttribute("aria-labelledby",this.$name.id),window.matchMedia("(pointer: coarse)").matches&&(this.$input.setAttribute("type","number"),this.$input.setAttribute("step","any")),this.$widget.appendChild(this.$input),this.$disable=this.$input;const t=()=>{let m=parseFloat(this.$input.value);isNaN(m)||(this._stepExplicit&&(m=this._snap(m)),this.setValue(this._clamp(m)))},i=m=>{const S=parseFloat(this.$input.value);isNaN(S)||(this._snapClampSetValue(S+m),this.$input.value=this.getValue())},s=m=>{m.key==="Enter"&&this.$input.blur(),m.code==="ArrowUp"&&(m.preventDefault(),i(this._step*this._arrowKeyMultiplier(m))),m.code==="ArrowDown"&&(m.preventDefault(),i(this._step*this._arrowKeyMultiplier(m)*-1))},a=m=>{this._inputFocused&&(m.preventDefault(),i(this._step*this._normalizeMouseWheel(m)))};let n=!1,o,l,c,p,u;const g=5,v=m=>{o=m.clientX,l=c=m.clientY,n=!0,p=this.getValue(),u=0,window.addEventListener("mousemove",w),window.addEventListener("mouseup",y)},w=m=>{if(n){const S=m.clientX-o,x=m.clientY-l;Math.abs(x)>g?(m.preventDefault(),this.$input.blur(),n=!1,this._setDraggingStyle(!0,"vertical")):Math.abs(S)>g&&y()}if(!n){const S=m.clientY-c;u-=S*this._step*this._arrowKeyMultiplier(m),p+u>this._max?u=this._max-p:p+u<this._min&&(u=this._min-p),this._snapClampSetValue(p+u)}c=m.clientY},y=()=>{this._setDraggingStyle(!1,"vertical"),this._callOnFinishChange(),window.removeEventListener("mousemove",w),window.removeEventListener("mouseup",y)},_=()=>{this._inputFocused=!0},f=()=>{this._inputFocused=!1,this.updateDisplay(),this._callOnFinishChange()};this.$input.addEventListener("input",t),this.$input.addEventListener("keydown",s),this.$input.addEventListener("wheel",a,{passive:!1}),this.$input.addEventListener("mousedown",v),this.$input.addEventListener("focus",_),this.$input.addEventListener("blur",f)}_initSlider(){this._hasSlider=!0,this.$slider=document.createElement("div"),this.$slider.classList.add("slider"),this.$fill=document.createElement("div"),this.$fill.classList.add("fill"),this.$slider.appendChild(this.$fill),this.$widget.insertBefore(this.$slider,this.$input),this.domElement.classList.add("hasSlider");const e=(f,m,S,x,T)=>(f-m)/(S-m)*(T-x)+x,t=f=>{const m=this.$slider.getBoundingClientRect();let S=e(f,m.left,m.right,this._min,this._max);this._snapClampSetValue(S)},i=f=>{this._setDraggingStyle(!0),t(f.clientX),window.addEventListener("mousemove",s),window.addEventListener("mouseup",a)},s=f=>{t(f.clientX)},a=()=>{this._callOnFinishChange(),this._setDraggingStyle(!1),window.removeEventListener("mousemove",s),window.removeEventListener("mouseup",a)};let n=!1,o,l;const c=f=>{f.preventDefault(),this._setDraggingStyle(!0),t(f.touches[0].clientX),n=!1},p=f=>{f.touches.length>1||(this._hasScrollBar?(o=f.touches[0].clientX,l=f.touches[0].clientY,n=!0):c(f),window.addEventListener("touchmove",u,{passive:!1}),window.addEventListener("touchend",g))},u=f=>{if(n){const m=f.touches[0].clientX-o,S=f.touches[0].clientY-l;Math.abs(m)>Math.abs(S)?c(f):(window.removeEventListener("touchmove",u),window.removeEventListener("touchend",g))}else f.preventDefault(),t(f.touches[0].clientX)},g=()=>{this._callOnFinishChange(),this._setDraggingStyle(!1),window.removeEventListener("touchmove",u),window.removeEventListener("touchend",g)},v=this._callOnFinishChange.bind(this),w=400;let y;const _=f=>{if(Math.abs(f.deltaX)<Math.abs(f.deltaY)&&this._hasScrollBar)return;f.preventDefault();const S=this._normalizeMouseWheel(f)*this._step;this._snapClampSetValue(this.getValue()+S),this.$input.value=this.getValue(),clearTimeout(y),y=setTimeout(v,w)};this.$slider.addEventListener("mousedown",i),this.$slider.addEventListener("touchstart",p,{passive:!1}),this.$slider.addEventListener("wheel",_,{passive:!1})}_setDraggingStyle(e,t="horizontal"){this.$slider&&this.$slider.classList.toggle("active",e),document.body.classList.toggle("lil-gui-dragging",e),document.body.classList.toggle(`lil-gui-${t}`,e)}_getImplicitStep(){return this._hasMin&&this._hasMax?(this._max-this._min)/1e3:.1}_onUpdateMinMax(){!this._hasSlider&&this._hasMin&&this._hasMax&&(this._stepExplicit||this.step(this._getImplicitStep(),!1),this._initSlider(),this.updateDisplay())}_normalizeMouseWheel(e){let{deltaX:t,deltaY:i}=e;return Math.floor(e.deltaY)!==e.deltaY&&e.wheelDelta&&(t=0,i=-e.wheelDelta/120,i*=this._stepExplicit?1:10),t+-i}_arrowKeyMultiplier(e){let t=this._stepExplicit?1:10;return e.shiftKey?t*=10:e.altKey&&(t/=10),t}_snap(e){let t=0;return this._hasMin?t=this._min:this._hasMax&&(t=this._max),e-=t,e=Math.round(e/this._step)*this._step,e+=t,e=parseFloat(e.toPrecision(15)),e}_clamp(e){return e<this._min&&(e=this._min),e>this._max&&(e=this._max),e}_snapClampSetValue(e){this.setValue(this._clamp(this._snap(e)))}get _hasScrollBar(){const e=this.parent.root.$children;return e.scrollHeight>e.clientHeight}get _hasMin(){return this._min!==void 0}get _hasMax(){return this._max!==void 0}}class Lt extends Y{constructor(e,t,i,s){super(e,t,i,"option"),this.$select=document.createElement("select"),this.$select.setAttribute("aria-labelledby",this.$name.id),this.$display=document.createElement("div"),this.$display.classList.add("display"),this.$select.addEventListener("change",()=>{this.setValue(this._values[this.$select.selectedIndex]),this._callOnFinishChange()}),this.$select.addEventListener("focus",()=>{this.$display.classList.add("focus")}),this.$select.addEventListener("blur",()=>{this.$display.classList.remove("focus")}),this.$widget.appendChild(this.$select),this.$widget.appendChild(this.$display),this.$disable=this.$select,this.options(s)}options(e){return this._values=Array.isArray(e)?e:Object.values(e),this._names=Array.isArray(e)?e:Object.keys(e),this.$select.replaceChildren(),this._names.forEach(t=>{const i=document.createElement("option");i.textContent=t,this.$select.appendChild(i)}),this.updateDisplay(),this}updateDisplay(){const e=this.getValue(),t=this._values.indexOf(e);return this.$select.selectedIndex=t,this.$display.textContent=t===-1?e:this._names[t],this}}class Ot extends Y{constructor(e,t,i){super(e,t,i,"string"),this.$input=document.createElement("input"),this.$input.setAttribute("type","text"),this.$input.setAttribute("spellcheck","false"),this.$input.setAttribute("aria-labelledby",this.$name.id),this.$input.addEventListener("input",()=>{this.setValue(this.$input.value)}),this.$input.addEventListener("keydown",s=>{s.code==="Enter"&&this.$input.blur()}),this.$input.addEventListener("blur",()=>{this._callOnFinishChange()}),this.$widget.appendChild(this.$input),this.$disable=this.$input,this.updateDisplay()}updateDisplay(){return this.$input.value=this.getValue(),this}}var kt=`.lil-gui {
  font-family: var(--font-family);
  font-size: var(--font-size);
  line-height: 1;
  font-weight: normal;
  font-style: normal;
  text-align: left;
  color: var(--text-color);
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
  --background-color: #1f1f1f;
  --text-color: #ebebeb;
  --title-background-color: #111111;
  --title-text-color: #ebebeb;
  --widget-color: #424242;
  --hover-color: #4f4f4f;
  --focus-color: #595959;
  --number-color: #2cc9ff;
  --string-color: #a2db3c;
  --font-size: 11px;
  --input-font-size: 11px;
  --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  --font-family-mono: Menlo, Monaco, Consolas, "Droid Sans Mono", monospace;
  --padding: 4px;
  --spacing: 4px;
  --widget-height: 20px;
  --title-height: calc(var(--widget-height) + var(--spacing) * 1.25);
  --name-width: 45%;
  --slider-knob-width: 2px;
  --slider-input-width: 27%;
  --color-input-width: 27%;
  --slider-input-min-width: 45px;
  --color-input-min-width: 45px;
  --folder-indent: 7px;
  --widget-padding: 0 0 0 3px;
  --widget-border-radius: 2px;
  --checkbox-size: calc(0.75 * var(--widget-height));
  --scrollbar-width: 5px;
}
.lil-gui, .lil-gui * {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}
.lil-gui.root {
  width: var(--width, 245px);
  display: flex;
  flex-direction: column;
  background: var(--background-color);
}
.lil-gui.root > .title {
  background: var(--title-background-color);
  color: var(--title-text-color);
}
.lil-gui.root > .children {
  overflow-x: hidden;
  overflow-y: auto;
}
.lil-gui.root > .children::-webkit-scrollbar {
  width: var(--scrollbar-width);
  height: var(--scrollbar-width);
  background: var(--background-color);
}
.lil-gui.root > .children::-webkit-scrollbar-thumb {
  border-radius: var(--scrollbar-width);
  background: var(--focus-color);
}
@media (pointer: coarse) {
  .lil-gui.allow-touch-styles, .lil-gui.allow-touch-styles .lil-gui {
    --widget-height: 28px;
    --padding: 6px;
    --spacing: 6px;
    --font-size: 13px;
    --input-font-size: 16px;
    --folder-indent: 10px;
    --scrollbar-width: 7px;
    --slider-input-min-width: 50px;
    --color-input-min-width: 65px;
  }
}
.lil-gui.force-touch-styles, .lil-gui.force-touch-styles .lil-gui {
  --widget-height: 28px;
  --padding: 6px;
  --spacing: 6px;
  --font-size: 13px;
  --input-font-size: 16px;
  --folder-indent: 10px;
  --scrollbar-width: 7px;
  --slider-input-min-width: 50px;
  --color-input-min-width: 65px;
}
.lil-gui.autoPlace {
  max-height: 100%;
  position: fixed;
  top: 0;
  right: 15px;
  z-index: 1001;
}

.lil-gui .controller {
  display: flex;
  align-items: center;
  padding: 0 var(--padding);
  margin: var(--spacing) 0;
}
.lil-gui .controller.disabled {
  opacity: 0.5;
}
.lil-gui .controller.disabled, .lil-gui .controller.disabled * {
  pointer-events: none !important;
}
.lil-gui .controller > .name {
  min-width: var(--name-width);
  flex-shrink: 0;
  white-space: pre;
  padding-right: var(--spacing);
  line-height: var(--widget-height);
}
.lil-gui .controller .widget {
  position: relative;
  display: flex;
  align-items: center;
  width: 100%;
  min-height: var(--widget-height);
}
.lil-gui .controller.string input {
  color: var(--string-color);
}
.lil-gui .controller.boolean {
  cursor: pointer;
}
.lil-gui .controller.color .display {
  width: 100%;
  height: var(--widget-height);
  border-radius: var(--widget-border-radius);
  position: relative;
}
@media (hover: hover) {
  .lil-gui .controller.color .display:hover:before {
    content: " ";
    display: block;
    position: absolute;
    border-radius: var(--widget-border-radius);
    border: 1px solid #fff9;
    top: 0;
    right: 0;
    bottom: 0;
    left: 0;
  }
}
.lil-gui .controller.color input[type=color] {
  opacity: 0;
  width: 100%;
  height: 100%;
  cursor: pointer;
}
.lil-gui .controller.color input[type=text] {
  margin-left: var(--spacing);
  font-family: var(--font-family-mono);
  min-width: var(--color-input-min-width);
  width: var(--color-input-width);
  flex-shrink: 0;
}
.lil-gui .controller.option select {
  opacity: 0;
  position: absolute;
  width: 100%;
  max-width: 100%;
}
.lil-gui .controller.option .display {
  position: relative;
  pointer-events: none;
  border-radius: var(--widget-border-radius);
  height: var(--widget-height);
  line-height: var(--widget-height);
  max-width: 100%;
  overflow: hidden;
  word-break: break-all;
  padding-left: 0.55em;
  padding-right: 1.75em;
  background: var(--widget-color);
}
@media (hover: hover) {
  .lil-gui .controller.option .display.focus {
    background: var(--focus-color);
  }
}
.lil-gui .controller.option .display.active {
  background: var(--focus-color);
}
.lil-gui .controller.option .display:after {
  font-family: "lil-gui";
  content: "↕";
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  padding-right: 0.375em;
}
.lil-gui .controller.option .widget,
.lil-gui .controller.option select {
  cursor: pointer;
}
@media (hover: hover) {
  .lil-gui .controller.option .widget:hover .display {
    background: var(--hover-color);
  }
}
.lil-gui .controller.number input {
  color: var(--number-color);
}
.lil-gui .controller.number.hasSlider input {
  margin-left: var(--spacing);
  width: var(--slider-input-width);
  min-width: var(--slider-input-min-width);
  flex-shrink: 0;
}
.lil-gui .controller.number .slider {
  width: 100%;
  height: var(--widget-height);
  background: var(--widget-color);
  border-radius: var(--widget-border-radius);
  padding-right: var(--slider-knob-width);
  overflow: hidden;
  cursor: ew-resize;
  touch-action: pan-y;
}
@media (hover: hover) {
  .lil-gui .controller.number .slider:hover {
    background: var(--hover-color);
  }
}
.lil-gui .controller.number .slider.active {
  background: var(--focus-color);
}
.lil-gui .controller.number .slider.active .fill {
  opacity: 0.95;
}
.lil-gui .controller.number .fill {
  height: 100%;
  border-right: var(--slider-knob-width) solid var(--number-color);
  box-sizing: content-box;
}

.lil-gui-dragging .lil-gui {
  --hover-color: var(--widget-color);
}
.lil-gui-dragging * {
  cursor: ew-resize !important;
}

.lil-gui-dragging.lil-gui-vertical * {
  cursor: ns-resize !important;
}

.lil-gui .title {
  height: var(--title-height);
  font-weight: 600;
  padding: 0 var(--padding);
  width: 100%;
  text-align: left;
  background: none;
  text-decoration-skip: objects;
}
.lil-gui .title:before {
  font-family: "lil-gui";
  content: "▾";
  padding-right: 2px;
  display: inline-block;
}
.lil-gui .title:active {
  background: var(--title-background-color);
  opacity: 0.75;
}
@media (hover: hover) {
  body:not(.lil-gui-dragging) .lil-gui .title:hover {
    background: var(--title-background-color);
    opacity: 0.85;
  }
  .lil-gui .title:focus {
    text-decoration: underline var(--focus-color);
  }
}
.lil-gui.root > .title:focus {
  text-decoration: none !important;
}
.lil-gui.closed > .title:before {
  content: "▸";
}
.lil-gui.closed > .children {
  transform: translateY(-7px);
  opacity: 0;
}
.lil-gui.closed:not(.transition) > .children {
  display: none;
}
.lil-gui.transition > .children {
  transition-duration: 300ms;
  transition-property: height, opacity, transform;
  transition-timing-function: cubic-bezier(0.2, 0.6, 0.35, 1);
  overflow: hidden;
  pointer-events: none;
}
.lil-gui .children:empty:before {
  content: "Empty";
  padding: 0 var(--padding);
  margin: var(--spacing) 0;
  display: block;
  height: var(--widget-height);
  font-style: italic;
  line-height: var(--widget-height);
  opacity: 0.5;
}
.lil-gui.root > .children > .lil-gui > .title {
  border: 0 solid var(--widget-color);
  border-width: 1px 0;
  transition: border-color 300ms;
}
.lil-gui.root > .children > .lil-gui.closed > .title {
  border-bottom-color: transparent;
}
.lil-gui + .controller {
  border-top: 1px solid var(--widget-color);
  margin-top: 0;
  padding-top: var(--spacing);
}
.lil-gui .lil-gui .lil-gui > .title {
  border: none;
}
.lil-gui .lil-gui .lil-gui > .children {
  border: none;
  margin-left: var(--folder-indent);
  border-left: 2px solid var(--widget-color);
}
.lil-gui .lil-gui .controller {
  border: none;
}

.lil-gui label, .lil-gui input, .lil-gui button {
  -webkit-tap-highlight-color: transparent;
}
.lil-gui input {
  border: 0;
  outline: none;
  font-family: var(--font-family);
  font-size: var(--input-font-size);
  border-radius: var(--widget-border-radius);
  height: var(--widget-height);
  background: var(--widget-color);
  color: var(--text-color);
  width: 100%;
}
@media (hover: hover) {
  .lil-gui input:hover {
    background: var(--hover-color);
  }
  .lil-gui input:active {
    background: var(--focus-color);
  }
}
.lil-gui input:disabled {
  opacity: 1;
}
.lil-gui input[type=text],
.lil-gui input[type=number] {
  padding: var(--widget-padding);
  -moz-appearance: textfield;
}
.lil-gui input[type=text]:focus,
.lil-gui input[type=number]:focus {
  background: var(--focus-color);
}
.lil-gui input[type=checkbox] {
  appearance: none;
  width: var(--checkbox-size);
  height: var(--checkbox-size);
  border-radius: var(--widget-border-radius);
  text-align: center;
  cursor: pointer;
}
.lil-gui input[type=checkbox]:checked:before {
  font-family: "lil-gui";
  content: "✓";
  font-size: var(--checkbox-size);
  line-height: var(--checkbox-size);
}
@media (hover: hover) {
  .lil-gui input[type=checkbox]:focus {
    box-shadow: inset 0 0 0 1px var(--focus-color);
  }
}
.lil-gui button {
  outline: none;
  cursor: pointer;
  font-family: var(--font-family);
  font-size: var(--font-size);
  color: var(--text-color);
  width: 100%;
  border: none;
}
.lil-gui .controller button {
  height: var(--widget-height);
  text-transform: none;
  background: var(--widget-color);
  border-radius: var(--widget-border-radius);
}
@media (hover: hover) {
  .lil-gui .controller button:hover {
    background: var(--hover-color);
  }
  .lil-gui .controller button:focus {
    box-shadow: inset 0 0 0 1px var(--focus-color);
  }
}
.lil-gui .controller button:active {
  background: var(--focus-color);
}

@font-face {
  font-family: "lil-gui";
  src: url("data:application/font-woff;charset=utf-8;base64,d09GRgABAAAAAAUsAAsAAAAACJwAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAABHU1VCAAABCAAAAH4AAADAImwmYE9TLzIAAAGIAAAAPwAAAGBKqH5SY21hcAAAAcgAAAD0AAACrukyyJBnbHlmAAACvAAAAF8AAACEIZpWH2hlYWQAAAMcAAAAJwAAADZfcj2zaGhlYQAAA0QAAAAYAAAAJAC5AHhobXR4AAADXAAAABAAAABMAZAAAGxvY2EAAANsAAAAFAAAACgCEgIybWF4cAAAA4AAAAAeAAAAIAEfABJuYW1lAAADoAAAASIAAAIK9SUU/XBvc3QAAATEAAAAZgAAAJCTcMc2eJxVjbEOgjAURU+hFRBK1dGRL+ALnAiToyMLEzFpnPz/eAshwSa97517c/MwwJmeB9kwPl+0cf5+uGPZXsqPu4nvZabcSZldZ6kfyWnomFY/eScKqZNWupKJO6kXN3K9uCVoL7iInPr1X5baXs3tjuMqCtzEuagm/AAlzQgPAAB4nGNgYRBlnMDAysDAYM/gBiT5oLQBAwuDJAMDEwMrMwNWEJDmmsJwgCFeXZghBcjlZMgFCzOiKOIFAB71Bb8AeJy1kjFuwkAQRZ+DwRAwBtNQRUGKQ8OdKCAWUhAgKLhIuAsVSpWz5Bbkj3dEgYiUIszqWdpZe+Z7/wB1oCYmIoboiwiLT2WjKl/jscrHfGg/pKdMkyklC5Zs2LEfHYpjcRoPzme9MWWmk3dWbK9ObkWkikOetJ554fWyoEsmdSlt+uR0pCJR34b6t/TVg1SY3sYvdf8vuiKrpyaDXDISiegp17p7579Gp3p++y7HPAiY9pmTibljrr85qSidtlg4+l25GLCaS8e6rRxNBmsnERunKbaOObRz7N72ju5vdAjYpBXHgJylOAVsMseDAPEP8LYoUHicY2BiAAEfhiAGJgZWBgZ7RnFRdnVJELCQlBSRlATJMoLV2DK4glSYs6ubq5vbKrJLSbGrgEmovDuDJVhe3VzcXFwNLCOILB/C4IuQ1xTn5FPilBTj5FPmBAB4WwoqAHicY2BkYGAA4sk1sR/j+W2+MnAzpDBgAyEMQUCSg4EJxAEAwUgFHgB4nGNgZGBgSGFggJMhDIwMqEAYAByHATJ4nGNgAIIUNEwmAABl3AGReJxjYAACIQYlBiMGJ3wQAEcQBEV4nGNgZGBgEGZgY2BiAAEQyQWEDAz/wXwGAAsPATIAAHicXdBNSsNAHAXwl35iA0UQXYnMShfS9GPZA7T7LgIu03SSpkwzYTIt1BN4Ak/gKTyAeCxfw39jZkjymzcvAwmAW/wgwHUEGDb36+jQQ3GXGot79L24jxCP4gHzF/EIr4jEIe7wxhOC3g2TMYy4Q7+Lu/SHuEd/ivt4wJd4wPxbPEKMX3GI5+DJFGaSn4qNzk8mcbKSR6xdXdhSzaOZJGtdapd4vVPbi6rP+cL7TGXOHtXKll4bY1Xl7EGnPtp7Xy2n00zyKLVHfkHBa4IcJ2oD3cgggWvt/V/FbDrUlEUJhTn/0azVWbNTNr0Ens8de1tceK9xZmfB1CPjOmPH4kitmvOubcNpmVTN3oFJyjzCvnmrwhJTzqzVj9jiSX911FjeAAB4nG3HMRKCMBBA0f0giiKi4DU8k0V2GWbIZDOh4PoWWvq6J5V8If9NVNQcaDhyouXMhY4rPTcG7jwYmXhKq8Wz+p762aNaeYXom2n3m2dLTVgsrCgFJ7OTmIkYbwIbC6vIB7WmFfAAAA==") format("woff");
}`;function Vt(r){const e=document.createElement("style");e.innerHTML=r;const t=document.querySelector("head link[rel=stylesheet], head style");t?document.head.insertBefore(e,t):document.head.appendChild(e)}let $e=!1;class Ue{constructor({parent:e,autoPlace:t=e===void 0,container:i,width:s,title:a="Controls",closeFolders:n=!1,injectStyles:o=!0,touchStyles:l=!0}={}){if(this.parent=e,this.root=e?e.root:this,this.children=[],this.controllers=[],this.folders=[],this._closed=!1,this._hidden=!1,this.domElement=document.createElement("div"),this.domElement.classList.add("lil-gui"),this.$title=document.createElement("button"),this.$title.classList.add("title"),this.$title.setAttribute("aria-expanded",!0),this.$title.addEventListener("click",()=>this.openAnimated(this._closed)),this.$title.addEventListener("touchstart",()=>{},{passive:!0}),this.$children=document.createElement("div"),this.$children.classList.add("children"),this.domElement.appendChild(this.$title),this.domElement.appendChild(this.$children),this.title(a),this.parent){this.parent.children.push(this),this.parent.folders.push(this),this.parent.$children.appendChild(this.domElement);return}this.domElement.classList.add("root"),l&&this.domElement.classList.add("allow-touch-styles"),!$e&&o&&(Vt(kt),$e=!0),i?i.appendChild(this.domElement):t&&(this.domElement.classList.add("autoPlace"),document.body.appendChild(this.domElement)),s&&this.domElement.style.setProperty("--width",s+"px"),this._closeFolders=n}add(e,t,i,s,a){if(Object(i)===i)return new Lt(this,e,t,i);const n=e[t];switch(typeof n){case"number":return new Ut(this,e,t,i,s,a);case"boolean":return new Ct(this,e,t);case"string":return new Ot(this,e,t);case"function":return new Ae(this,e,t)}console.error(`gui.add failed
	property:`,t,`
	object:`,e,`
	value:`,n)}addColor(e,t,i=1){return new Dt(this,e,t,i)}addFolder(e){const t=new Ue({parent:this,title:e});return this.root._closeFolders&&t.close(),t}load(e,t=!0){return e.controllers&&this.controllers.forEach(i=>{i instanceof Ae||i._name in e.controllers&&i.load(e.controllers[i._name])}),t&&e.folders&&this.folders.forEach(i=>{i._title in e.folders&&i.load(e.folders[i._title])}),this}save(e=!0){const t={controllers:{},folders:{}};return this.controllers.forEach(i=>{if(!(i instanceof Ae)){if(i._name in t.controllers)throw new Error(`Cannot save GUI with duplicate property "${i._name}"`);t.controllers[i._name]=i.save()}}),e&&this.folders.forEach(i=>{if(i._title in t.folders)throw new Error(`Cannot save GUI with duplicate folder "${i._title}"`);t.folders[i._title]=i.save()}),t}open(e=!0){return this._setClosed(!e),this.$title.setAttribute("aria-expanded",!this._closed),this.domElement.classList.toggle("closed",this._closed),this}close(){return this.open(!1)}_setClosed(e){this._closed!==e&&(this._closed=e,this._callOnOpenClose(this))}show(e=!0){return this._hidden=!e,this.domElement.style.display=this._hidden?"none":"",this}hide(){return this.show(!1)}openAnimated(e=!0){return this._setClosed(!e),this.$title.setAttribute("aria-expanded",!this._closed),requestAnimationFrame(()=>{const t=this.$children.clientHeight;this.$children.style.height=t+"px",this.domElement.classList.add("transition");const i=a=>{a.target===this.$children&&(this.$children.style.height="",this.domElement.classList.remove("transition"),this.$children.removeEventListener("transitionend",i))};this.$children.addEventListener("transitionend",i);const s=e?this.$children.scrollHeight:0;this.domElement.classList.toggle("closed",!e),requestAnimationFrame(()=>{this.$children.style.height=s+"px"})}),this}title(e){return this._title=e,this.$title.textContent=e,this}reset(e=!0){return(e?this.controllersRecursive():this.controllers).forEach(i=>i.reset()),this}onChange(e){return this._onChange=e,this}_callOnChange(e){this.parent&&this.parent._callOnChange(e),this._onChange!==void 0&&this._onChange.call(this,{object:e.object,property:e.property,value:e.getValue(),controller:e})}onFinishChange(e){return this._onFinishChange=e,this}_callOnFinishChange(e){this.parent&&this.parent._callOnFinishChange(e),this._onFinishChange!==void 0&&this._onFinishChange.call(this,{object:e.object,property:e.property,value:e.getValue(),controller:e})}onOpenClose(e){return this._onOpenClose=e,this}_callOnOpenClose(e){this.parent&&this.parent._callOnOpenClose(e),this._onOpenClose!==void 0&&this._onOpenClose.call(this,e)}destroy(){this.parent&&(this.parent.children.splice(this.parent.children.indexOf(this),1),this.parent.folders.splice(this.parent.folders.indexOf(this),1)),this.domElement.parentElement&&this.domElement.parentElement.removeChild(this.domElement),Array.from(this.children).forEach(e=>e.destroy())}controllersRecursive(){let e=Array.from(this.controllers);return this.folders.forEach(t=>{e=e.concat(t.controllersRecursive())}),e}foldersRecursive(){let e=Array.from(this.folders);return this.folders.forEach(t=>{e=e.concat(t.foldersRecursive())}),e}}var le=function(){var r=0,e=document.createElement("div");e.style.cssText="position:fixed;top:0;left:0;cursor:pointer;opacity:0.9;z-index:10000",e.addEventListener("click",function(p){p.preventDefault(),i(++r%e.children.length)},!1);function t(p){return e.appendChild(p.dom),p}function i(p){for(var u=0;u<e.children.length;u++)e.children[u].style.display=u===p?"block":"none";r=p}var s=(performance||Date).now(),a=s,n=0,o=t(new le.Panel("FPS","#0ff","#002")),l=t(new le.Panel("MS","#0f0","#020"));if(self.performance&&self.performance.memory)var c=t(new le.Panel("MB","#f08","#201"));return i(0),{REVISION:16,dom:e,addPanel:t,showPanel:i,begin:function(){s=(performance||Date).now()},end:function(){n++;var p=(performance||Date).now();if(l.update(p-s,200),p>a+1e3&&(o.update(n*1e3/(p-a),100),a=p,n=0,c)){var u=performance.memory;c.update(u.usedJSHeapSize/1048576,u.jsHeapSizeLimit/1048576)}return p},update:function(){s=this.end()},domElement:e,setMode:i}};le.Panel=function(r,e,t){var i=1/0,s=0,a=Math.round,n=a(window.devicePixelRatio||1),o=80*n,l=48*n,c=3*n,p=2*n,u=3*n,g=15*n,v=74*n,w=30*n,y=document.createElement("canvas");y.width=o,y.height=l,y.style.cssText="width:80px;height:48px";var _=y.getContext("2d");return _.font="bold "+9*n+"px Helvetica,Arial,sans-serif",_.textBaseline="top",_.fillStyle=t,_.fillRect(0,0,o,l),_.fillStyle=e,_.fillText(r,c,p),_.fillRect(u,g,v,w),_.fillStyle=t,_.globalAlpha=.9,_.fillRect(u,g,v,w),{dom:y,update:function(f,m){i=Math.min(i,f),s=Math.max(s,f),_.fillStyle=t,_.globalAlpha=1,_.fillRect(0,0,o,g),_.fillStyle=e,_.fillText(a(f)+" "+r+" ("+a(i)+"-"+a(s)+")",c,p),_.drawImage(y,u+n,g,v-n,w,u,g,v-n,w),_.fillRect(u+v-n,g,n,w),_.fillStyle=t,_.globalAlpha=.9,_.fillRect(u+v-n,g,n,a((1-f/m)*w))}}};const O={MOVE:0,BUMP:1,ERODE:2},X={CONSTANT:0,LINEAR:1,SMOOTH:2,GAUSSIAN:3,SHARP:4};class It{constructor(){this.state={operation:O.MOVE,falloff:X.SMOOTH,radius:.3,strength:.8,displacement:new Float32Array(3),targetValue:0},this.isActive=!1,this.modifiers={shift:!1,alt:!1,ctrl:!1},this.moveStartPos=null,this.moveStartWorldPos=null,this.moveAccumulated=new Float32Array(3),this.sculptingState=null}setOperation(e){this.state.operation=e}setFalloff(e){this.state.falloff=e}setSculptingState(e){this.sculptingState=e}setRadius(e){this.state.radius=Math.max(.01,e)}setStrength(e){this.state.strength=Math.max(0,Math.min(1,e))}setModifiers(e,t,i){this.modifiers.shift=e,this.modifiers.alt=t,this.modifiers.ctrl=i}startMove(e,t){this.moveStartPos=e,this.moveStartWorldPos=[...t],this.moveAccumulated.fill(0)}updateMove(e,t,i,s,a,n,o){if(!this.moveStartPos||this.state.operation!==O.MOVE)return;const l={x:e[0]-this.moveStartPos[0],y:e[1]-this.moveStartPos[1]};let c;if(s&&a&&n&&o?c=s(l,a,n,o):c=[l.x*.01,-l.y*.01,0],this.state.displacement[0]=c[0],this.state.displacement[1]=c[1],this.state.displacement[2]=c[2],this.sculptingState){const p=[this.moveStartWorldPos[0]+c[0],this.moveStartWorldPos[1]+c[1],this.moveStartWorldPos[2]+c[2]];this.sculptingState.updateBrushSettings("position",{position:p})}}endMove(){this.moveStartPos=null,this.moveStartWorldPos=null,this.state.displacement.fill(0)}getGPUParams(){var i;const e=this.sculptingState?this.sculptingState.getBrushSettings():{position:new Float32Array([0,0,0]),normal:new Float32Array([0,1,0])},t=((i=d.sculpting.brushes)==null?void 0:i.erode)||{strengthBias:.15};return{position:e.position,radius:this.state.radius,strength:this.state.strength,operation:this.state.operation,falloffType:this.state.falloff,normal:e.normal,targetValue:this.state.targetValue,erodeBias:t.strengthBias,grabOriginalPos:this.moveStartWorldPos}}getOperationName(){return{[O.MOVE]:"Move",[O.BUMP]:"Bump",[O.ERODE]:"Erode"}[this.state.operation]||"Unknown"}getFalloffName(){return{[X.CONSTANT]:"Constant",[X.LINEAR]:"Linear",[X.ERODE]:"Erode",[X.GAUSSIAN]:"Gaussian",[X.SHARP]:"Sharp"}[this.state.falloff]||"Unknown"}}class Ht{constructor(e){this.viewport=e,this.gui=new Ue({title:"STUFF"}),this.gui.close(),this.setupSculptingFolder(),this.setupCameraFolder(),this.setupRenderingFolder(),this.statsFps=le(),this.statsFps.showPanel(0),document.body.appendChild(this.statsFps.dom)}statsBegin(){this.statsFps&&this.statsFps.begin()}statsEnd(){this.statsFps&&this.statsFps.end()}setupRenderingFolder(){const e=this.gui.addFolder("Rendering");this.viewport.settings;const t=d.raymarching;e.add(t,"maxSteps",16,256,1).name("Max Steps").onChange(()=>this.viewport.isDirty=!0),e.add(d.camera,"farPlane",10,998).name("Max Distance").onChange(()=>this.viewport.isDirty=!0),e.add(t,"surfaceEpsilon",1e-4,1,1e-4).name("Surface Precision").onChange(()=>this.viewport.isDirty=!0);const i=e.addFolder("Adaptive Ray Marching"),s=t.adaptive;i.add(s,"enabled").name("Enable Adaptive Quality").onChange(()=>this.viewport.isDirty=!0),i.add(s,"qualityMultiplier",.1,3,.1).name("Quality Multiplier").onChange(()=>this.viewport.isDirty=!0),i.add(s,"complexityThreshold",.1,1,.1).name("Complexity Threshold").onChange(()=>this.viewport.isDirty=!0),i.add(s,"distanceScale",.01,1,.01).name("Distance Scale").onChange(()=>this.viewport.isDirty=!0),i.add(s,"minStepMultiplier",.01,1,.01).name("Min Step Size").onChange(()=>this.viewport.isDirty=!0),i.add(s,"maxStepMultiplier",1,500,1).name("Max Step Size").onChange(()=>this.viewport.isDirty=!0)}setupCameraFolder(){const e=this.gui.addFolder("Camera Intrinsics");this.setupSensorPresets(e),this.setupFocalLength(e),this.setupLensDistortion(e)}setupSensorPresets(e){const t={"Full Frame (36x24mm)":{width:36,height:24},"APS-C Canon (22.3x14.9mm)":{width:22.3,height:14.9},"APS-C Nikon/Sony (23.6x15.6mm)":{width:23.6,height:15.6},"Micro 4/3 (17.3x13mm)":{width:17.3,height:13},"Super 35 (24.9x18.7mm)":{width:24.9,height:18.7},'1" Sensor (13.2x8.8mm)':{width:13.2,height:8.8},'1/1.7" (7.6x5.7mm)':{width:7.6,height:5.7},'1/2.3" (6.17x4.55mm)':{width:6.17,height:4.55},"iPhone 15 Pro (9.8x7.3mm)":{width:9.8,height:7.3},Custom:{width:d.camera.sensorWidth,height:d.camera.sensorHeight}},i=()=>{const s=Math.sqrt(d.camera.sensorWidth*d.camera.sensorWidth+d.camera.sensorHeight*d.camera.sensorHeight),n=2*Math.atan(s/(2*d.camera.focalLength))*180/Math.PI;this.fovDisplay.value=Math.round(n*10)/10,this.fovController.updateDisplay()};e.add({preset:"Full Frame (36x24mm)"},"preset",Object.keys(t)).name("Sensor Preset").onChange(s=>{const a=t[s];a&&s!=="Custom"&&(d.camera.sensorWidth=a.width,d.camera.sensorHeight=a.height,this.viewport.camera.isDirty=!0,this.viewport.isDirty=!0,this.sensorWidthController.updateDisplay(),this.sensorHeightController.updateDisplay(),i())}),this.sensorWidthController=e.add(d.camera,"sensorWidth",1,50).name("Sensor Width (mm)").onChange(s=>{this.viewport.camera.isDirty=!0,this.viewport.isDirty=!0,i(),L.emit(k.CAMERA_UPDATE,{source:"sensorWidth",sensorWidth:s})}),this.sensorHeightController=e.add(d.camera,"sensorHeight",1,50).name("Sensor Height (mm)").onChange(s=>{this.viewport.camera.isDirty=!0,this.viewport.isDirty=!0,i(),L.emit(k.CAMERA_UPDATE,{source:"sensorHeight",sensorHeight:s})}),this.fovDisplay={value:0},this.fovController=e.add(this.fovDisplay,"value",0,180).name("Field of View (°)").disable(),this.updateFOV=i,i()}setupFocalLength(e){const t=e.add(d.camera,"focalLength",8,400).name("Focal Length (mm)").onChange(s=>{this.viewport.camera.isDirty=!0,this.viewport.isDirty=!0,this.updateFOV(),L.emit(k.CAMERA_UPDATE,{source:"focalLength",focalLength:s})}),i={"14mm (Ultra Wide)":14,"24mm (Wide)":24,"35mm (Wide Normal)":35,"50mm (Normal)":50,"85mm (Portrait)":85,"135mm (Telephoto)":135,"200mm (Telephoto)":200,"300mm (Super Telephoto)":300,Custom:d.camera.focalLength};e.add({preset:"50mm (Normal)"},"preset",Object.keys(i)).name("Focal Length Preset").onChange(s=>{const a=i[s];a&&s!=="Custom"&&(d.camera.focalLength=a,this.viewport.camera.isDirty=!0,this.viewport.isDirty=!0,t.updateDisplay(),this.updateFOV())})}setupLensDistortion(e){const t=this.viewport.settings,i=d.camera;e.add(t,"lensDistortionEnabled").name("Enable Distortion").onChange(()=>this.viewport.isDirty=!0);const s=e.addFolder("Distortion Parameters");s.add(i,"radialK1",-1,1).name("Radial K1").onChange(()=>this.viewport.isDirty=!0),s.add(i,"radialK2",-1,1).name("Radial K2").onChange(()=>this.viewport.isDirty=!0),s.add(i,"tangentialP1",-.5,.5).name("Tangential P1").onChange(()=>this.viewport.isDirty=!0),s.add(i,"tangentialP2",-.5,.5).name("Tangential P2").onChange(()=>this.viewport.isDirty=!0),s.add(i,"principalPointX",0,1).name("Principal X").onChange(()=>{this.viewport.camera.isDirty=!0,this.viewport.isDirty=!0}),s.add(i,"principalPointY",0,1).name("Principal Y").onChange(()=>{this.viewport.camera.isDirty=!0,this.viewport.isDirty=!0}),e.add({preset:()=>this.applyDistortionPreset("barrel")},"preset").name("Apply Barrel Preset"),e.add({preset:()=>this.applyDistortionPreset("pincushion")},"preset").name("Apply Pincushion Preset"),e.add({reset:()=>this.applyDistortionPreset("reset")},"reset").name("Reset Distortion")}setupSculptingFolder(){const e=this.gui.addFolder("Sculpting"),t=this.viewport.sculptingSystem;t&&(this.sculptSettings={brushType:"Move",brushRadius:.3,brushStrength:.8},e.add(this.sculptSettings,"brushType",["Move","Bump","Erode"]).name("Brush Type").onChange(i=>{const s={Move:O.MOVE,Bump:O.BUMP,Erode:O.ERODE};t.setBrushType(s[i]);const a=t.brushManager.state;this.sculptSettings.brushRadius=a.radius,this.sculptSettings.brushStrength=a.strength,this.radiusController&&this.radiusController.updateDisplay(),this.strengthController&&this.strengthController.updateDisplay()}),this.radiusController=e.add(this.sculptSettings,"brushRadius",.01,.5,.01).name("Radius").onChange(i=>t.setBrushRadius(i)),this.strengthController=e.add(this.sculptSettings,"brushStrength",.1,1,.1).name("Strength").onChange(i=>t.setBrushStrength(i)),e.open())}applyDistortionPreset(e){const t=this.viewport.settings,i=d.camera,a={barrel:{radialK1:.1,radialK2:.05,tangentialP1:0,tangentialP2:0,principalPointX:.5,principalPointY:.5,enabled:!0},pincushion:{radialK1:-.15,radialK2:-.05,tangentialP1:0,tangentialP2:0,principalPointX:.5,principalPointY:.5,enabled:!0},reset:{radialK1:0,radialK2:0,tangentialP1:0,tangentialP2:0,principalPointX:.5,principalPointY:.5,enabled:!1}}[e];a&&(Object.assign(i,{radialK1:a.radialK1,radialK2:a.radialK2,tangentialP1:a.tangentialP1,tangentialP2:a.tangentialP2,principalPointX:a.principalPointX,principalPointY:a.principalPointY}),t.lensDistortionEnabled=a.enabled),this.viewport.isDirty=!0}destroy(){this.gui&&(this.gui.destroy(),this.gui=null),this.statsFps&&(document.body.removeChild(this.statsFps.dom),this.statsFps=null)}}const b={BUFFER:"buffer",TEXTURE:"texture",SAMPLER:"sampler",STORAGE_TEXTURE:"storageTexture"};class Nt{constructor(e){this.device=e,this.commonSamplers=new Map}createBuffer(e){return this.device.createBuffer(e)}createUniformBuffer(e,t,i={}){return this.createBuffer({label:e||`uniformBuffer_${t}bytes`,size:t,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,...i})}createStorageBuffer(e,t,i=!1){let s=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST;return i&&(s|=GPUBufferUsage.COPY_SRC),this.createBuffer({label:e||`storageBuffer_${t}bytes`,size:t,usage:s})}createVertexBuffer(e,t){return this.createBuffer({label:e||`vertexBuffer_${t}bytes`,size:t,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST})}createTexture(e){return this.device.createTexture(e)}createRenderTexture(e,t,i,s="rgba8unorm",a={}){return this.createTexture({label:e||`renderTexture_${t}x${i}_${s}`,size:{width:t,height:i,depthOrArrayLayers:1},format:s,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,...a})}createDepthTexture(e,t,i,s="depth24plus"){return this.createTexture({label:e||`depthTexture_${t}x${i}_${s}`,size:{width:t,height:i,depthOrArrayLayers:1},format:s,usage:GPUTextureUsage.RENDER_ATTACHMENT})}createStorageTexture(e,t,i,s="rgba8unorm"){return this.createTexture({label:e||`storageTexture_${t}x${i}_${s}`,size:{width:t,height:i,depthOrArrayLayers:1},format:s,usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING})}createSampler(e={}){return this.device.createSampler(e)}createLinearSampler(e="linearSampler",t={}){return this.createSampler({label:e,magFilter:"linear",minFilter:"linear",mipmapFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge",...t})}createNearestSampler(e="nearestSampler",t={}){return this.createSampler({label:e,magFilter:"nearest",minFilter:"nearest",mipmapFilter:"nearest",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge",...t})}getLinearSampler(){return this.commonSamplers.has("linear")||this.commonSamplers.set("linear",this.createLinearSampler("common linear sampler")),this.commonSamplers.get("linear")}getNearestSampler(){return this.commonSamplers.has("nearest")||this.commonSamplers.set("nearest",this.createNearestSampler("common nearest sampler")),this.commonSamplers.get("nearest")}destroyResource(e){e&&e.destroy&&e.destroy()}destroy(){this.commonSamplers.clear()}}let ie=null;function $t(r){return ie||(ie=new Nt(r),ie)}function he(){if(!ie)throw new Error("[ResourceManager] Global resource manager not initialized. Call initializeGlobalResourceManager first.");return ie}class Yt{constructor(e,t,i){this.device=e,this.width=t,this.height=i,this.textures=new Map,this.views=new Map,this.textureConfigs={color:{format:"rgba16float",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,label:"G-Buffer Color"},position:{format:"rgba16float",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC,label:"G-Buffer Position"},normal:{format:"rgba16float",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC,label:"G-Buffer Normal"},depth:{format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT,label:"G-Buffer Depth"},material:{format:"rgba8unorm",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC,label:"G-Buffer Material Properties",optional:!0}},this.readbackBufferPool=new Map,this.readbackBuffersInUse=new Map,this.readbackPending=!1,this.stats={totalMemory:0,resizeCount:0,lastResizeTime:0},this.createTextures()}createTextures(){this.destroyTextures(),this.stats.totalMemory=0;for(const[e,t]of Object.entries(this.textureConfigs)){if(t.optional&&!this.isTextureNeeded(e))continue;const s=he().createTexture({size:{width:this.width,height:this.height,depthOrArrayLayers:1},format:t.format,usage:t.usage,label:t.label});this.textures.set(e,s),this.views.set(e,s.createView({label:`${t.label} View`}));const a=this.getBytesPerPixel(t.format),n=this.width*this.height*a;this.stats.totalMemory+=n}}isTextureNeeded(e){return!1}getBytesPerPixel(e){return{rgba8unorm:4,rgba16float:8,rgba32float:16,depth24plus:4,depth32float:4,r32float:4,rg32float:8}[e]||4}resize(e,t){return e===this.width&&t===this.height?!1:(this.width=e,this.height=t,this.stats.resizeCount++,this.stats.lastResizeTime=performance.now(),this.createTextures(),this.clearReadbackBuffers(),L.emit("gbuffer-textures-recreated",{width:this.width,height:this.height,textures:this.textures,views:this.views}),!0)}getTextureView(e){return this.views.get(e)}getTexture(e){return this.textures.get(e)}getAllViews(){const e={};for(const[t,i]of this.views)e[t]=i;return e}getPostProcessingTextures(){return{colorTexture:this.getTexture("color"),positionTexture:this.getTexture("position"),normalTexture:this.getTexture("normal"),depthTexture:this.getTexture("depth")}}getRenderPassDescriptor(){return this.createRenderPassDescriptor([0,0,0,0])}createRenderPassDescriptor(e=[0,0,0,0]){const t=[],i=["color","position","normal","material"];for(const a of i){const n=this.views.get(a);n&&t.push({view:n,clearValue:e,loadOp:"clear",storeOp:"store"})}const s=this.views.get("depth");return{colorAttachments:t,depthStencilAttachment:s?{view:s,depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}:void 0}}async readPixel(e,t,i="position"){const s=this.textures.get(i);if(!s)return null;e=Math.max(0,Math.min(this.width-1,Math.floor(e))),t=Math.max(0,Math.min(this.height-1,Math.floor(t)));let a=null;const n=i;let o=this.readbackBufferPool.get(n);if(o||(o=[],this.readbackBufferPool.set(n,o)),a=o.pop(),!a){const u=this.getBytesPerPixel(this.textureConfigs[i].format);a=he().createBuffer({size:Math.max(u,256),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,label:`Readback buffer for ${i}`})}const l=`${i}_${e}_${t}_${Date.now()}`;this.readbackBuffersInUse.set(l,a);const c=this.device.createCommandEncoder();return c.copyTextureToBuffer({texture:s,origin:[e,t,0]},{buffer:a,bytesPerRow:256,rowsPerImage:1},[1,1,1]),this.device.queue.submit([c.finish()]),await a.mapAsync(GPUMapMode.READ).then(()=>{const u=new Float32Array(a.getMappedRange()),g=Array.from(u);a.unmap(),this.readbackBuffersInUse.delete(l);const v=this.readbackBufferPool.get(n);return v&&v.length<5&&v.push(a),g})}clearReadbackBuffers(){const e=he();for(const t of this.readbackBuffersInUse.values())e.destroyResource(t);this.readbackBuffersInUse.clear();for(const t of this.readbackBufferPool.values())for(const i of t)e.destroyResource(i);this.readbackBufferPool.clear()}getStats(){return{...this.stats,textureCount:this.textures.size,dimensions:`${this.width}x${this.height}`}}validate(){const e=[];for(const[s,a]of Object.entries(this.textureConfigs))!a.optional&&!this.textures.has(s)&&e.push(`Missing required texture: ${s}`);(this.width<=0||this.height<=0)&&e.push(`Invalid dimensions: ${this.width}x${this.height}`);const t=512,i=this.stats.totalMemory/1024/1024;return i>t&&e.push(`Memory usage (${i.toFixed(2)} MB) exceeds limit (${t} MB)`),{valid:e.length===0,issues:e}}destroyTextures(){const e=he();for(const t of this.textures.values())e.destroyResource(t);this.textures.clear(),this.views.clear()}destroy(){this.destroyTextures(),this.clearReadbackBuffers(),this.stats.totalMemory=0}}class qt{constructor(e){if(!e||!(e instanceof Ze))throw new Error("RenderPipelineManager requires a valid WebGPUContext instance");this.webgpu=e,this.device=e.device,this.pipelines=new Map,this.shaderModules=new Map,this.stats={totalCompilations:0,cacheHits:0,compilationErrors:0,averageCompilationTime:0},this.lastError=null,this.errorHandlers=[]}async createPipeline(e){const t=performance.now();if(this.validateDescriptor(e),this.pipelines.has(e.name)){this.stats.cacheHits++;const u=this.pipelines.get(e.name);return u.usageCount++,u}const i=await this.createShaderModule(e.vertexShader,`${e.name} Vertex Shader`),s=await this.createShaderModule(e.fragmentShader,`${e.name} Fragment Shader`),a=this.createBindGroupLayouts(e.bindGroupLayouts),n=this.device.createPipelineLayout({label:`${e.name} Pipeline Layout`,bindGroupLayouts:a}),o={label:e.name,layout:n,vertex:{module:i,entryPoint:e.vertexEntryPoint||"vs_main",...e.vertexState},fragment:{module:s,entryPoint:e.fragmentEntryPoint||"fs_main",targets:e.fragmentState.targets},primitive:e.primitiveState||{topology:"triangle-list"}};e.depthStencilState&&(o.depthStencil=e.depthStencilState),e.multisampleState&&(o.multisample=e.multisampleState);const l=this.device.createRenderPipeline(o),c=performance.now()-t,p={pipeline:l,bindGroupLayouts:a,layout:n,compilationTime:c,createdAt:new Date,usageCount:1,descriptor:{...e}};return this.pipelines.set(e.name,p),this.stats.totalCompilations++,this.updateAverageCompilationTime(c),e.onCompile&&e.onCompile(p),p}async createShaderModule(e,t){const i=this.hashCode(e);if(this.shaderModules.has(i))return this.shaderModules.get(i);const s=await this.webgpu.createShaderModule({label:t,code:e});return this.shaderModules.set(i,s),s}createBindGroupLayouts(e){return e.map((t,i)=>this.device.createBindGroupLayout({label:t.label||`Bind Group Layout ${i}`,entries:t.entries}))}getPipeline(e){const t=this.pipelines.get(e);return t&&t.usageCount++,t||null}validatePipelineCompatibility(e){const t=this.pipelines.get(e);return t?!this.device||this.device.lost?{valid:!1,error:"WebGPU device lost"}:{valid:!0,pipeline:t.pipeline}:{valid:!1,error:"Pipeline not found"}}getStatistics(){const e=[];for(const[t,i]of this.pipelines)e.push({name:t,compilationTime:i.compilationTime,createdAt:i.createdAt,usageCount:i.usageCount});return{...this.stats,pipelines:e,cacheSize:this.pipelines.size,shaderCacheSize:this.shaderModules.size}}clearCache(e=null){e?this.pipelines.delete(e):(this.pipelines.clear(),this.clearShaderCache())}clearShaderCache(){this.shaderModules.clear()}addErrorHandler(e){this.errorHandlers.push(e)}removeErrorHandler(e){const t=this.errorHandlers.indexOf(e);t!==-1&&this.errorHandlers.splice(t,1)}validateDescriptor(e){if(!e.name)throw new Error("Pipeline descriptor must have a name");if(!e.vertexShader)throw new Error("Pipeline descriptor must have vertex shader code");if(!e.fragmentShader)throw new Error("Pipeline descriptor must have fragment shader code");if(!e.fragmentState||!e.fragmentState.targets)throw new Error("Pipeline descriptor must have fragment state with targets");if(!e.bindGroupLayouts||!Array.isArray(e.bindGroupLayouts))throw new Error("Pipeline descriptor must have bindGroupLayouts array")}notifyErrorHandlers(e,t){for(const i of this.errorHandlers)i(e,t)}updateAverageCompilationTime(e){const t=this.stats.totalCompilations,i=this.stats.averageCompilationTime;this.stats.averageCompilationTime=(i*(t-1)+e)/t}hashCode(e){let t=0;for(let i=0;i<e.length;i++){const s=e.charCodeAt(i);t=(t<<5)-t+s,t=t&t}return t}destroy(){this.pipelines.clear(),this.shaderModules.clear(),this.errorHandlers=[]}}class Wt{constructor(e){this.device=e,this.voxelSize=d.voxel.voxelSize,this.hashTableSize=d.voxel.hashTableSize,this.maxVoxels=d.voxel.maxVoxels,this.voxelDataBuffer=null,this.hashTableBuffer=null,this.paramsBuffer=null,this.voxelAllocatorBuffer=null,this.readBindGroup=null,this.writeBindGroup=null,this.isInitialized=!1,this.initPromise=null,this.initPromise=this.initialize(),this.currentVoxelCount=0}async initialize(){if(this.isInitialized)return;this.voxelDataBuffer=this.device.createBuffer({label:"Voxel Data Buffer",size:this.maxVoxels*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.hashTableBuffer=this.device.createBuffer({label:"Voxel Hash Table",size:this.hashTableSize*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.paramsBuffer=this.device.createBuffer({label:"VoxelHashMap Parameters",size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.voxelAllocatorBuffer=this.device.createBuffer({label:"Voxel Allocator",size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});const e=new Int32Array(this.hashTableSize*4),t=2147483647;for(let a=0;a<this.hashTableSize;a++)e[a*4]=t,e[a*4+1]=t,e[a*4+2]=t,e[a*4+3]=-1;this.device.queue.writeBuffer(this.hashTableBuffer,0,e);const i=new Uint32Array(8),s=new Float32Array(i.buffer);s[0]=this.voxelSize,i[1]=this.hashTableSize,i[2]=this.maxVoxels,i[3]=0,i[4]=8,i[5]=0,i[6]=0,i[7]=0,this.device.queue.writeBuffer(this.paramsBuffer,0,i),this.device.queue.writeBuffer(this.voxelAllocatorBuffer,0,new Uint32Array([0])),this.isInitialized=!0}async waitForInit(){return this.initPromise&&await this.initPromise,this.isInitialized}getMemoryStats(){return{voxelCount:this.currentVoxelCount,maxVoxels:this.maxVoxels,usagePercent:this.currentVoxelCount/this.maxVoxels*100,memoryMB:this.currentVoxelCount*4/(1024*1024),hashTableFillPercent:this.currentVoxelCount/this.hashTableSize*100}}async updateVoxelCount(){const e=this.device.createBuffer({size:4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),t=this.device.createCommandEncoder();t.copyBufferToBuffer(this.voxelAllocatorBuffer,0,e,0,4),this.device.queue.submit([t.finish()]),await e.mapAsync(GPUMapMode.READ);const i=new Uint32Array(e.getMappedRange())[0];return e.unmap(),e.destroy(),this.currentVoxelCount=i,i}destroy(){this.voxelDataBuffer&&(this.voxelDataBuffer.destroy(),this.voxelDataBuffer=null),this.hashTableBuffer&&(this.hashTableBuffer.destroy(),this.hashTableBuffer=null),this.voxelAllocatorBuffer&&(this.voxelAllocatorBuffer.destroy(),this.voxelAllocatorBuffer=null),this.paramsBuffer&&(this.paramsBuffer.destroy(),this.paramsBuffer=null),this.isInitialized=!1}getReadBindGroup(e){var t;return e?this.createFullReadBindGroup(e):(!this.readBindGroup&&((t=this.device.bindGroupLayouts)!=null&&t.voxelRead)&&(this.readBindGroup=this.createFullReadBindGroup(this.device.bindGroupLayouts.voxelRead)),this.readBindGroup)}getWriteBindGroup(e){var t;return e?this.createWriteBindGroup(e):(!this.writeBindGroup&&((t=this.device.bindGroupLayouts)!=null&&t.voxelWrite)&&(this.writeBindGroup=this.createWriteBindGroup(this.device.bindGroupLayouts.voxelWrite)),this.writeBindGroup)}createFullReadBindGroup(e){return this.device.createBindGroup({label:"VoxelHashMap Full Read Bind Group",layout:e,entries:[{binding:0,resource:{buffer:this.hashTableBuffer}},{binding:1,resource:{buffer:this.voxelDataBuffer}},{binding:2,resource:{buffer:this.paramsBuffer}}]})}createWriteBindGroup(e){return this.device.createBindGroup({label:"VoxelHashMap Write Bind Group",layout:e,entries:[{binding:0,resource:{buffer:this.hashTableBuffer}},{binding:1,resource:{buffer:this.voxelDataBuffer}},{binding:2,resource:{buffer:this.paramsBuffer}}]})}createBindGroup(e){var a;const t=new Uint32Array(8),i=new Float32Array(t.buffer);return i[0]=this.voxelSize,t[1]=this.hashTableSize,t[2]=this.maxVoxels,t[3]=0,t[4]=8,t[5]=0,t[6]=0,t[7]=0,this.device.queue.writeBuffer(this.paramsBuffer,0,t),((a=e.entries)==null?void 0:a.length)===5?this.device.createBindGroup({layout:e,entries:[{binding:0,resource:{buffer:this.hashTableBuffer}},{binding:1,resource:{buffer:this.voxelDataBuffer}},{binding:2,resource:{buffer:this.paramsBuffer}},{binding:3,resource:{buffer:this.paramsBuffer}},{binding:4,resource:{buffer:this.voxelAllocatorBuffer}}]}):this.device.createBindGroup({layout:e,entries:[{binding:0,resource:{buffer:this.hashTableBuffer}},{binding:1,resource:{buffer:this.voxelDataBuffer}},{binding:2,resource:{buffer:this.paramsBuffer}}]})}addTestGeometry(){const e=new Int32Array(this.hashTableSize*4),t=new Float32Array(this.maxVoxels),i=2147483647;for(let x=0;x<this.hashTableSize;x++)e[x*4]=i,e[x*4+1]=i,e[x*4+2]=i,e[x*4+3]=-1;let s=0;const a=(x,T,A,E)=>{if(s>=this.maxVoxels)return;const C=2538058380;let z=C^x;z=Math.imul(z,3432918353),z=z<<15|z>>>17,z=Math.imul(z,461845907);let F=C^T;F=Math.imul(F,3432918353),F=F<<15|F>>>17,F=Math.imul(F,461845907);let B=C^A;B=Math.imul(B,3432918353),B=B<<15|B>>>17,B=Math.imul(B,461845907);let P=C;P^=z,P=(P<<13|P>>>19)+3864292196,P^=F,P=(P<<13|P>>>19)+3864292196,P^=B,P^=P>>>16,P=Math.imul(P,2246822507),P^=P>>>13,P=Math.imul(P,3266489909),P^=P>>>16;let H=(P>>>0)%this.hashTableSize;for(let q=0;q<256;q++){const I=H*4;if(e[I+3]===-1){e[I]=x,e[I+1]=T,e[I+2]=A,e[I+3]=s,t[s]=E,s++;break}if(e[I]===x&&e[I+1]===T&&e[I+2]===A){t[e[I+3]]=E;break}H=(H+1)%this.hashTableSize}},n=(x,T,A,E,C,z,F)=>{const B=x-E,P=T-C,V=A-z;return Math.sqrt(B*B+P*P+V*V)-F},o=(x,T,A,E,C,z,F,B,P)=>{const V=Math.abs(x-E)-F,H=Math.abs(T-C)-B,q=Math.abs(A-z)-P;return Math.sqrt(Math.max(V,0)**2+Math.max(H,0)**2+Math.max(q,0)**2)+Math.min(Math.max(V,Math.max(H,q)),0)},l=(x,T,A,E,C,z,F,B)=>{const P=x-E,V=T-C,H=A-z,q=Math.sqrt(P*P+H*H)-F;return Math.sqrt(q*q+V*V)-B},p=5*this.voxelSize,u=[-1.5,-.5,-.5],g=[1.375,.5,.5],v=p,w=[Math.floor((u[0]-v)/this.voxelSize),Math.floor((u[1]-v)/this.voxelSize),Math.floor((u[2]-v)/this.voxelSize)],y=[Math.ceil((g[0]+v)/this.voxelSize),Math.ceil((g[1]+v)/this.voxelSize),Math.ceil((g[2]+v)/this.voxelSize)],_=[{type:"sphere",center:[0,0,0],radius:.5},{type:"box",center:[1,0,0],size:[.375,.375,.375]},{type:"torus",center:[-1,0,0],R:.375,r:.125}];for(let x=w[0];x<=y[0];x++)for(let T=w[1];T<=y[1];T++)for(let A=w[2];A<=y[2];A++){const E=(x+.5)*this.voxelSize,C=(T+.5)*this.voxelSize,z=(A+.5)*this.voxelSize;let F=1e3;for(const B of _){let P=0;B.type==="sphere"?P=n(E,C,z,B.center[0],B.center[1],B.center[2],B.radius):B.type==="box"?P=o(E,C,z,B.center[0],B.center[1],B.center[2],B.size[0],B.size[1],B.size[2]):B.type==="torus"&&(P=l(E,C,z,B.center[0],B.center[1],B.center[2],B.R,B.r)),F=Math.min(F,P)}Math.abs(F)<p&&a(x,T,A,F)}this.device.queue.writeBuffer(this.hashTableBuffer,0,e),this.device.queue.writeBuffer(this.voxelDataBuffer,0,t);const f=new Uint32Array(8),m=new Float32Array(f.buffer);m[0]=this.voxelSize,f[1]=this.hashTableSize,f[2]=this.maxVoxels,f[3]=s,f[4]=8,f[5]=0,f[6]=0,f[7]=0,this.device.queue.writeBuffer(this.paramsBuffer,0,f),this.device.queue.writeBuffer(this.voxelAllocatorBuffer,0,new Uint32Array([s])),this.currentVoxelCount=s;const S=[[0,0,0],[1,0,0],[-1,0,0]];for(const[x,T,A]of S){const E=[(x+.5)*this.voxelSize,(T+.5)*this.voxelSize,(A+.5)*this.voxelSize];for(const C of _)C.type==="sphere"&&n(E[0],E[1],E[2],C.center[0],C.center[1],C.center[2],C.radius)}}}class Xt{constructor(e){this.limits=e}getLinearWorkgroupSize(){const e=this.limits.maxInvocations,t=this.limits.maxX,i=[256,128,64,32];for(const s of i)if(s<=t&&s<=e)return[s,1,1];return[Math.min(t,e),1,1]}get2DWorkgroupSize(){const e=this.limits.maxInvocations,t=this.limits.maxX,i=this.limits.maxY,s=[[16,16],[8,8],[32,8],[16,8]];for(const[n,o]of s)if(n<=t&&o<=i&&n*o<=e)return[n,o,1];const a=Math.floor(Math.sqrt(e));return[Math.min(a,t),Math.min(a,i),1]}get3DWorkgroupSize(){const e=this.limits.maxInvocations,t=this.limits.maxX,i=this.limits.maxY,s=this.limits.maxZ,a=[[8,8,4],[4,4,4],[8,4,4],[4,4,8]];for(const[o,l,c]of a)if(o<=t&&l<=i&&c<=s&&o*l*c<=e)return[o,l,c];const n=Math.floor(Math.cbrt(e));return[Math.min(n,t),Math.min(n,i),Math.min(n,s)]}getOptimalWorkgroupSize(e="linear"){switch(e){case"3d":case"voxel":return this.get3DWorkgroupSize();case"2d":case"image":return this.get2DWorkgroupSize();case"linear":case"1d":default:return this.getLinearWorkgroupSize()}}calculateDispatchSize(e,t){return Array.isArray(e)?e.map((i,s)=>Math.ceil(i/(t[s]||1))):Math.ceil(e/t[0])}}function jt(r){return`@workgroup_size(${r[0]}, ${r[1]}, ${r[2]})`}const Ye=(r=[1,1,1])=>`
${U.metadata}
${U.constants}
${De}
${Pe}
${We}
${Xe}

// VoxelHashMap bindings for query operations (read-only)
@group(1) @binding(0) var<storage, read> voxel_hash_table: array<vec4<i32>>;
@group(1) @binding(1) var<storage, read> voxel_sdf_data: array<f32>;
@group(1) @binding(2) var<storage, read> voxel_params_buffer: VoxelHashMapParams;

// Adaptive cache bindings (group 2)
@group(2) @binding(0) var<storage, read> cache_data: array<f32>;
@group(2) @binding(1) var<uniform> cache_metadata: CacheMetadata;

struct RaycastParams {
    origin: vec3<f32>,
    _padding1: f32,
    direction: vec3<f32>,
    _padding2: f32,
    max_distance: f32,
    epsilon: f32,
    brush_preview_epsilon: f32,
    use_brush_preview_mode: f32,
}

struct RaycastResult {
    hit: f32,
    distance: f32,
    _padding1: f32,
    _padding2: f32,
    position: vec3<f32>,
    _padding3: f32,
    normal: vec3<f32>,
    _padding4: f32,
}

// Uniforms for adaptive marching (kept for compatibility)
struct Uniforms {
    surface_epsilon: f32,
    adaptive_enabled: f32,
    adaptive_quality_multiplier: f32,
    adaptive_min_step_multiplier: f32,
    adaptive_max_step_multiplier: f32,
    adaptive_distance_scale: f32,
}

@group(0) @binding(0) var<uniform> params: RaycastParams;
@group(0) @binding(1) var<storage, read_write> result: RaycastResult;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

// Query SDF with adaptive cache integration
fn query_sdf_with_cache(world_pos: vec3<f32>) -> f32 {
    // Check if adaptive cache is active
    if (cache_metadata.is_active > 0u) {
        // Try to sample from adaptive cache
        let cache_value = query_adaptive_cache(
            world_pos,
            &cache_data,
            cache_metadata.origin,
            cache_metadata.dimensions,
            cache_metadata.voxel_size
        );
        
        // If we got a valid value from cache (not empty space marker)
        if (cache_value < 9.0) {
            return cache_value;
        }
    }
    
    // Fallback to voxel hash map
    return query_voxel_hashmap(world_pos, voxel_params_buffer.voxel_size, &voxel_hash_table, &voxel_sdf_data);
}

// Define query_sdf alias for the shared robust ray marcher
fn query_sdf(pos: vec3<f32>) -> f32 {
    return query_sdf_with_cache(pos);
}

// INTERACTIVE PERFORMANCE: Fast surface intersection optimized for sculpting responsiveness
fn raycast_surface(ray_origin: vec3<f32>, ray_dir: vec3<f32>, max_dist: f32, epsilon: f32) -> RaycastResult {
    var res: RaycastResult;
    res.hit = 0.0;
    res.distance = max_dist;
    res.position = vec3<f32>(0.0);
    res.normal = vec3<f32>(0.0, 1.0, 0.0);
    
    let voxel_size = voxel_params_buffer.voxel_size;
    
    // PERFORMANCE FIX: Reduced step count for interactive sculpting operations
    // Sculpting needs fast response more than perfect precision
    let max_steps = 256; // Halved from 512 for responsiveness

    let hit = trace_ray(
        ray_origin, 
        ray_dir, 
        max_steps, 
        max_dist,
        voxel_size
    );
    
    if (hit.hit) {
        res.hit = 1.0;
        res.distance = hit.distance;
        res.position = hit.position;
        
        // PERFORMANCE FIX: Faster normal calculation for interactive operations
        // Use larger epsilon and simpler calculation for speed
        let eps = voxel_size * 0.7; // Larger epsilon = fewer SDF queries = faster response
        let dx = query_sdf_with_cache(res.position + vec3<f32>(eps, 0.0, 0.0)) - query_sdf_with_cache(res.position - vec3<f32>(eps, 0.0, 0.0));
        let dy = query_sdf_with_cache(res.position + vec3<f32>(0.0, eps, 0.0)) - query_sdf_with_cache(res.position - vec3<f32>(0.0, eps, 0.0));
        let dz = query_sdf_with_cache(res.position + vec3<f32>(0.0, 0.0, eps)) - query_sdf_with_cache(res.position - vec3<f32>(0.0, 0.0, eps));
        res.normal = normalize(vec3<f32>(dx, dy, dz));
    }
    
    return res;
}

@compute ${jt(r)}
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let origin = params.origin;
    let direction = normalize(params.direction);
    let epsilon = select(params.epsilon, params.brush_preview_epsilon, params.use_brush_preview_mode > 0.5);
    
    // Perform raycast
    result = raycast_surface(origin, direction, params.max_distance, epsilon);
    
}
`;class Kt{constructor(e,t){this.device=e,this.workgroupSizes=t}initializePipelines(){const e=this.createBindGroupLayouts(),t=this.createPipelines(e);return{layouts:e,pipelines:t}}createBindGroupLayouts(){const e=this.device.createBindGroupLayout({label:"VoxelHashMap Compute Bind Group Layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),t=this.device.createBindGroupLayout({label:"VoxelHashMap Read-Only Bind Group Layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}]}),i=this.device.createBindGroupLayout({label:"Brush Parameters Bind Group Layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),s=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]});return{voxelHashMap:e,voxelHashMapReadOnly:t,brushParams:i,raycast:s}}createPipelines(e){const t=this.device.createShaderModule({label:"GPU Raycast Shader",code:Ye([1,1,1])}),i=this.device.createBindGroupLayout({label:"Raycast Cache Bind Group Layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]});return{raycast:this.device.createComputePipeline({label:"GPU Raycast Pipeline",layout:this.device.createPipelineLayout({bindGroupLayouts:[e.raycast,e.voxelHashMapReadOnly,i]}),compute:{module:t,entryPoint:"main"}})}}createCacheAwareRaycastPipeline(e,t){const i=this.device.createShaderModule({label:"GPU Raycast Shader (Cache-Aware)",code:Ye([1,1,1])});return this.device.createComputePipeline({label:"GPU Raycast Pipeline (Cache-Aware)",layout:this.device.createPipelineLayout({bindGroupLayouts:[e.raycast,e.voxelHashMapReadOnly,t]}),compute:{module:i,entryPoint:"main"}})}}class Jt{constructor(e){this.device=e,this.buffers=new Map}initializeBuffers(){this.buffers.set("brushParams",this.device.createBuffer({label:"Brush Parameters Buffer",size:d.gpu.bufferSizes.brushParameters,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})),this.buffers.set("raycastParams",this.device.createBuffer({label:"Raycast Parameters Buffer",size:d.gpu.bufferSizes.raycastParameters,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})),this.buffers.set("raycastResult",this.device.createBuffer({label:"Raycast Result Buffer",size:d.gpu.bufferSizes.raycastResult,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC})),this.buffers.set("raycastStaging",this.device.createBuffer({label:"Raycast Staging Buffer",size:d.gpu.bufferSizes.raycastResult,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST})),this.buffers.set("raycastUniforms",this.device.createBuffer({label:"Raycast Uniforms Buffer",size:d.gpu.bufferSizes.sculptingUniforms,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})),this.updateRaycastUniforms()}updateBrushParameters(e){const t=this.buffers.get("brushParams");(!t||t.size<d.gpu.bufferSizes.brushParameters)&&(t&&t.destroy(),this.buffers.set("brushParams",this.device.createBuffer({label:"Brush Parameters Buffer",size:d.gpu.bufferSizes.brushParameters,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})));const i=new ArrayBuffer(64),s=new DataView(i);s.setFloat32(0,e.position[0],!0),s.setFloat32(4,e.position[1],!0),s.setFloat32(8,e.position[2],!0),s.setFloat32(12,e.radius,!0),s.setFloat32(16,e.strength,!0),s.setUint32(20,e.operation||O.MOVE,!0),s.setUint32(24,e.falloffType||X.ERODE,!0),s.setFloat32(28,e.targetValue||0,!0),s.setFloat32(32,e.normal[0],!0),s.setFloat32(36,e.normal[1],!0),s.setFloat32(40,e.normal[2],!0),s.setFloat32(44,e.erodeBias,!0),s.setFloat32(48,e.grabOriginalPos?e.grabOriginalPos[0]:e.position[0],!0),s.setFloat32(52,e.grabOriginalPos?e.grabOriginalPos[1]:e.position[1],!0),s.setFloat32(56,e.grabOriginalPos?e.grabOriginalPos[2]:e.position[2],!0),s.setUint32(60,0,!0),this.device.queue.writeBuffer(this.buffers.get("brushParams"),0,i)}updateRaycastParameters(e,t,i,s,a=null,n=!1){const o=new Float32Array(16);o[0]=e[0],o[1]=e[1],o[2]=e[2],o[3]=0,o[4]=t[0],o[5]=t[1],o[6]=t[2],o[7]=0,o[8]=i,o[9]=s,o[10]=a,o[11]=n?1:0,this.device.queue.writeBuffer(this.buffers.get("raycastParams"),0,o)}updateRaycastUniforms(){const e=new Float32Array(8);e[0]=d.raymarching.surfaceEpsilon,e[1]=d.raymarching.adaptive.enabled?1:0,e[2]=d.raymarching.adaptive.qualityMultiplier,e[3]=d.raymarching.adaptive.minStepMultiplier,e[4]=d.raymarching.adaptive.maxStepMultiplier,e[5]=d.raymarching.adaptive.distanceScale,e[6]=0,e[7]=0,this.device.queue.writeBuffer(this.buffers.get("raycastUniforms"),0,e)}getBuffer(e){return this.buffers.get(e)}async readRaycastResult(){const e=this.buffers.get("raycastStaging");await e.mapAsync(GPUMapMode.READ);const t=new Float32Array(e.getMappedRange()),i=t[0]>.5,s=t[1],a=[t[4],t[5],t[6]],n=[t[8],t[9],t[10]];return e.unmap(),i?{position:a,normal:n,distance:s}:null}destroy(){for(const e of this.buffers.values())e&&e.destroy();this.buffers.clear()}}const se={UNIFORMS:0,SPATIAL_DATA:1,MATERIAL:2};class Zt{constructor(e){this.device=e,this.definitions=new Map,this.layouts=new Map}defineBindGroup(e,t,i){this.definitions.set(e,{name:e,slot:t,entries:i,resources:new Map,bindGroup:null,isDirty:!0})}registerLayout(e,t){this.layouts.set(e,t)}setResource(e,t,i,s=b.BUFFER){const a=this.definitions.get(e);a&&(a.resources.set(t,{resource:i,type:s}),a.isDirty=!0)}setResources(e,t){const i=this.definitions.get(e);if(i){for(const[s,{resource:a,type:n}]of Object.entries(t))i.resources.set(parseInt(s),{resource:a,type:n||b.BUFFER});i.isDirty=!0}}getBindGroup(e,t=null){const i=this.definitions.get(e);if(!i)throw new Error(`No definition found for bind group ${e}`);if(i.bindGroup&&!i.isDirty)return i.bindGroup;const s=t?this.layouts.get(t):null;if(!s)throw new Error(`No layout found for ${t||e}`);i.resources||(i.resources=new Map);const a=[];for(const n of i.entries){const o=i.resources.get(n.binding);if(!o)throw new Error(`Missing resource for binding ${n.binding} in ${e}`);a.push({binding:n.binding,resource:this.createResourceBinding(o)})}return i.bindGroup=this.device.createBindGroup({label:`${e} Bind Group`,layout:s,entries:a}),i.isDirty=!1,i.bindGroup}createResourceBinding(e){const{resource:t,type:i}=e;switch(i){case b.BUFFER:return{buffer:t};case b.TEXTURE:return t.createView?t.createView():t;case b.SAMPLER:return t;case b.STORAGE_TEXTURE:return t.createView?t.createView():t;default:throw new Error(`Unknown resource type: ${i}`)}}invalidate(e){const t=this.definitions.get(e);t&&(t.isDirty=!0)}destroy(){this.definitions.clear(),this.layouts.clear()}}const Ce={uniform:{label:"Uniforms Bind Group Layout",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}}]},voxelHashMap:{label:"Voxel Hash Map Bind Group Layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]},adaptiveCache:{label:"Adaptive Cache Bind Group Layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]}};class Qt{constructor(e){this.bindGroupSystem=e,this.layouts=null}setLayouts(e){this.layouts=e}initializeBindGroups(){this.bindGroupSystem.defineBindGroup("sculpting_voxel_compute",se.SPATIAL_DATA,[{binding:0,type:b.BUFFER},{binding:1,type:b.BUFFER},{binding:2,type:b.BUFFER}]),this.bindGroupSystem.defineBindGroup("sculpting_voxel_readonly",se.SPATIAL_DATA,[{binding:0,type:b.BUFFER},{binding:1,type:b.BUFFER},{binding:2,type:b.BUFFER}]),this.bindGroupSystem.defineBindGroup("sculpting_brush_params",se.MATERIAL,[{binding:0,type:b.BUFFER}]),this.bindGroupSystem.defineBindGroup("sculpting_raycast",se.UNIFORMS,[{binding:0,type:b.BUFFER},{binding:1,type:b.BUFFER},{binding:2,type:b.BUFFER}]),this.layouts&&(this.layouts.voxelHashMap&&this.bindGroupSystem.registerLayout("sculpting_voxel_compute_layout",this.layouts.voxelHashMap),this.layouts.voxelHashMapReadOnly&&this.bindGroupSystem.registerLayout("sculpting_voxel_readonly_layout",this.layouts.voxelHashMapReadOnly),this.layouts.brushParams&&this.bindGroupSystem.registerLayout("sculpting_brush_params_layout",this.layouts.brushParams),this.layouts.raycast&&this.bindGroupSystem.registerLayout("sculpting_raycast_layout",this.layouts.raycast))}refreshBindGroups(e,t){e&&(this.bindGroupSystem.setResources("sculpting_voxel_compute",{0:{resource:e.hashTableBuffer,type:b.BUFFER},1:{resource:e.voxelDataBuffer,type:b.BUFFER},2:{resource:e.paramsBuffer,type:b.BUFFER}}),this.bindGroupSystem.setResources("sculpting_voxel_readonly",{0:{resource:e.hashTableBuffer,type:b.BUFFER},1:{resource:e.voxelDataBuffer,type:b.BUFFER},2:{resource:e.paramsBuffer,type:b.BUFFER}}));const i=t==null?void 0:t.getBuffer("brushParams");i&&this.bindGroupSystem.setResource("sculpting_brush_params",0,i,b.BUFFER);const s=t==null?void 0:t.getBuffer("raycastParams"),a=t==null?void 0:t.getBuffer("raycastResult"),n=t==null?void 0:t.getBuffer("raycastUniforms");s&&a&&n&&this.bindGroupSystem.setResources("sculpting_raycast",{0:{resource:s,type:b.BUFFER},1:{resource:a,type:b.BUFFER},2:{resource:n,type:b.BUFFER}})}getBindGroup(e){const i={voxelHashMap:"sculpting_voxel_compute",voxelHashMapReadOnly:"sculpting_voxel_readonly",brushParams:"sculpting_brush_params",raycast:"sculpting_raycast"}[e],s=i+"_layout";return this.bindGroupSystem.getBindGroup(i,s)}}class ei{constructor(){this.initializeBrushSettings(),this.initializeStateTracking()}initializeBrushSettings(){this.brushSettings={radius:.3,strength:.8,position:new Float32Array([0,0,0]),normal:new Float32Array([0,1,0]),falloffType:X.ERODE},this._tempPosition=new Float32Array(3),this._tempNormal=new Float32Array(3)}initializeStateTracking(){this.showBrushPreview=!1,this.brushPreviewPosition=[0,0,0],this.lastRaycastHit=null,this.lastRayOrigin=null,this.lastRayDirection=null,this.pendingSculptOperation=null,this.isProcessingSculpt=!1,this.sculptOperationCount=0,this.needsBindGroupRefresh=!0}updateBrushSettings(e,t){switch(e){case"radius":this.brushSettings.radius=t.radius;break;case"strength":this.brushSettings.strength=t.strength;break;case"position":t.position&&this.brushSettings.position.set(t.position);break;case"normal":t.normal&&this.brushSettings.normal.set(t.normal);break;case"type":this.brushSettings.type=t.type;break}}getBrushSettings(){return this.brushSettings}setPendingOperation(e){this.pendingSculptOperation=e}clearPendingOperation(){this.pendingSculptOperation=null}setProcessingState(e){this.isProcessingSculpt=e}incrementOperationCount(){this.sculptOperationCount++}}class ti{constructor(e,t,i){this.device=e,this.editingCache=t,this.brushManager=i,this.pendingSyncTimer=null}async execute(e,t){const{viewport:i,bufferManager:s,bindGroupManager:a,voxelHashMap:n}=t;L.emit(k.SCULPT_START,e),s.updateBrushParameters(e),await this.performSculptingCompute(e,t),this.handlePostSculptingTasks(i)}async performSculptingCompute(e,t){const i=d.sculpting.adaptiveCache.maxSculptingDistance,s=t.viewport.camera.position;if(Math.sqrt(Math.pow(e.position[0]-s[0],2)+Math.pow(e.position[1]-s[1],2)+Math.pow(e.position[2]-s[2],2))>i)return;const{viewport:n,bindGroupManager:o,voxelHashMap:l}=t;if(o.getBindGroup("brushParams"),(!this.editingCache.fillPipeline||!this.editingCache.fillPipeline.pipeline)&&await this.editingCache.createPipelines(this.brushManager),this.editingCache.isActive)await this.editingCache.updateBrushPosition(e.position);else{const p=this.brushManager.getOperationName(),u=d.camera,g=n.canvas.width/n.canvas.height,v=u.sensorHeight,w=2*Math.atan(v/(2*u.focalLength)),y={position:n.camera.position,fov:w,aspect:g,near:u.nearPlane,far:u.farPlane,viewMatrix:n.camera.viewMatrix,projectionMatrix:n.camera.projectionMatrix},_={width:n.canvas.width,height:n.canvas.height};if(!await this.editingCache.allocateCacheForBrush(p,e.position,e.radius,y,_))throw new Error(`Failed to allocate cache for brush operation: ${p} at position [${e.position}] with radius ${e.radius}`);n.camera.update(n.canvas.width,n.canvas.height);const m={view:n.camera.viewMatrix,projection:n.camera.projectionMatrix};this.editingCache.activate(m),await this.editingCache.fillFromVoxels(l),await this.editingCache.applyInitialRedistancing(),n.sculptingSystem&&n.sculptingSystem.needsSnapshotCapture&&(await this.editingCache.captureSnapshot(),n.sculptingSystem.needsSnapshotCapture=!1),n.isDirty=!0}const c={position:e.position,radius:e.radius,strength:e.strength,type:this.brushManager.getOperationName(),falloffType:this.brushManager.state.falloff,normal:e.normal,grabOriginalPos:e.grabOriginalPos};await this.editingCache.applyBrush(c),n.isDirty=!0}handlePostSculptingTasks(e){e.isDirty=!0}async syncEditingCache(e,t){!this.editingCache.isActive||!this.editingCache.needsSync||(this.editingCache.onBrushRelease(),this.pendingSyncTimer&&clearTimeout(this.pendingSyncTimer),this.pendingSyncTimer=setTimeout(async()=>{try{const i=await this.editingCache.syncToVoxels(e);await this.device.queue.onSubmittedWorkDone(),await this.editingCache.deactivate(),t.isDirty=!0}catch(i){console.error("Error syncing editing cache:",i)}},0))}}class ii{constructor(e,t,i,s){this.device=e,this.bufferManager=t,this.sculptingSystem=i,this.pipelines=s,this.pendingOperation=null,this.isProcessing=!1,this.lastRaycastTime=0,this.viewport=null,this.hasCacheAwarePipeline=!1,this.cachedGBufferData=null,this.gBufferReadbackInProgress=!1}setViewport(e){this.viewport=e}async performRaycast(e,t,i=!1){if(this.shouldUseGBuffer()){const s=await this.tryGBufferRaycast();if(s)return s}return new Promise((s,a)=>{this.pendingOperation={origin:[...e],direction:[...t],forBrushPreview:i,resolve:s,reject:a},this.isProcessing||this.processNextRaycast()})}shouldUseGBuffer(){return d.sculpting.alwaysCalculateGBufferNormals}async tryGBufferRaycast(){const e=this.viewport.input.mouse;if(this.cachedGBufferData){const t=this.cachedGBufferData;return this.cachedGBufferData=null,this.gBufferReadbackInProgress||(this.gBufferReadbackInProgress=!0,this.requestGBufferReadbackAsync(e)),t}return this.gBufferReadbackInProgress||(this.gBufferReadbackInProgress=!0,this.requestGBufferReadbackAsync(e)),null}requestGBufferReadbackAsync(e){Promise.all([this.viewport.gBufferManager.readPixel(e.x,e.y,"position"),this.viewport.gBufferManager.readPixel(e.x,e.y,"normal")]).then(([t,i])=>{t&&i&&i[3]>.5&&(this.cachedGBufferData={position:[t[0],t[1],t[2]],normal:[i[0],i[1],i[2]],distance:t[3]}),this.gBufferReadbackInProgress=!1}).catch(()=>{this.gBufferReadbackInProgress=!1})}async processNextRaycast(){if(this.pendingOperation){for(this.isProcessing=!0;this.pendingOperation;){const e=this.pendingOperation;this.pendingOperation=null;const t=await this.executeGPURaycast(e.origin,e.direction,e.forBrushPreview);e.resolve(t)}this.isProcessing=!1}}async executeGPURaycast(e,t,i=!1){var w,y;const s=d.camera.farPlane,a=d.raymarching.surfaceEpsilon,n=i?d.raymarching.brushPreviewEpsilon:null;this.bufferManager.updateRaycastParameters(e,t,s,a,n,i);const o=this.device.createCommandEncoder(),l=o.beginComputePass();if(l.setPipeline(this.pipelines.raycast),!((y=(w=this.sculptingSystem)==null?void 0:w.voxelHashMap)!=null&&y.isInitialized))return l.end(),null;const c=this.sculptingSystem.getBindGroup("raycast"),p=this.sculptingSystem.getBindGroup("voxelHashMapReadOnly");l.setBindGroup(0,c),l.setBindGroup(1,p);const u=this.sculptingSystem.editingCache;let g=null;if(u&&u.adaptiveCache&&u.adaptiveCache.dataBindGroup){const _=this.pipelines.raycast.getBindGroupLayout(2);u.adaptiveCache.raycastBindGroup||u.adaptiveCache.createRaycastCompatibleBindGroup(_),g=u.adaptiveCache.raycastBindGroup}else g=this.getOrCreateDummyCacheBindGroup();return g&&l.setBindGroup(2,g),l.dispatchWorkgroups(1,1,1),l.end(),o.copyBufferToBuffer(this.bufferManager.getBuffer("raycastResult"),0,this.bufferManager.getBuffer("raycastStaging"),0,64),this.device.queue.submit([o.finish()]),await this.bufferManager.readRaycastResult()}getOrCreateDummyCacheBindGroup(){if(!this.dummyCacheBindGroup){const e=this.device.createBuffer({label:"Dummy Cache Data",size:d.bufferSizes.MIN_DUMMY_CACHE_SIZE,usage:GPUBufferUsage.STORAGE}),t=this.device.createBuffer({label:"Dummy Cache Metadata",size:d.bufferSizes.CACHE_METADATA_SIZE,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),i=new ArrayBuffer(d.bufferSizes.CACHE_METADATA_SIZE),s=new Float32Array(i),a=new Uint32Array(i);s.fill(0),s[3]=.1,a[7]=0,this.device.queue.writeBuffer(t,0,i);const n=this.pipelines.raycast.getBindGroupLayout(2);this.dummyCacheBindGroup=this.device.createBindGroup({label:"Dummy Cache Bind Group",layout:n,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:t}}]})}return this.dummyCacheBindGroup}updateLastRaycastTime(){this.lastRaycastTime=performance.now()}}class si{constructor(e,t){if(this.viewport=e,this.sculptingSystem=t,this.input=e.input,!this.input)throw new Error("[SculptingInputHandler] Viewport must have input system initialized!");this.isSculpting=!1,this.isNavigating=!1,this.modifierKeys={ctrl:!1,shift:!1,alt:!1,meta:!1},this.previousBrushType=null,this.isShiftBrushActive=!1,this.tabletPressure=d.sculpting.input.defaultTabletPressure,this.pointerType="mouse",this.lastMouseMoveTime=0,this.mouseThrottleFPS=d.sculpting.input.mouseThrottleFPS,this.mouseThrottleInterval=this.mouseThrottleFPS>0?1e3/this.mouseThrottleFPS:0,this.lastSculptTime=0,this.sculptThrottleInterval=16,this.setupHandlers()}setupHandlers(){this.input.mapMouseDown(0,"sculpt:start"),this.input.mapMouseUp(0,"sculpt:end"),this.input.mapMouseMove("sculpt:hover"),this.input.onAction("sculpt:start",e=>this.handleSculptStart(e),{priority:10}),this.input.onAction("camera:rotate",e=>this.handleCameraRotate(e),{priority:1}),this.input.onAction("sculpt:end",()=>this.handleSculptEnd()),this.input.onAction("sculpt:hover",e=>this.handleSculptHover(e)),window.addEventListener("keydown",e=>this.handleKeyDown(e)),window.addEventListener("keyup",e=>this.handleKeyUp(e))}handleSculptStart(e){var t,i;if(this.sculptingSystem.showBrushPreview&&this.sculptingSystem.lastRaycastHit){const s=this.sculptingSystem.state.getBrushSettings();if(this.pointerType=((t=e.event)==null?void 0:t.pointerType)||"mouse",this.tabletPressure=(i=e.event)==null?void 0:i.pressure,this.tabletPressure===void 0&&(this.tabletPressure=d.sculpting.input.defaultTabletPressure,this.pointerType),this.isSculpting=!0,this.sculptingSystem.lastRaycastHit&&this.sculptingSystem.lastRaycastHit.position&&this.sculptingSystem.lastRaycastHit.normal&&(s.position.set(this.sculptingSystem.lastRaycastHit.position),s.normal.set(this.sculptingSystem.lastRaycastHit.normal)),this.sculptingSystem.brushManager.state.operation===O.MOVE){const a=[e.mouse.x,e.mouse.y],n=this.sculptingSystem.lastRaycastHit.position;this.sculptingSystem.brushManager.startMove(a,n),this.sculptingSystem.needsSnapshotCapture=!0}return this.sculptingSystem.brushManager.state.operation===O.MOVE?(L.emit(k.SCULPT_START,{position:s.position,radius:s.radius,operation:this.sculptingSystem.brushManager.state.operation,modifiers:e.modifiers}),this.sculptingSystem.performSculpting(),this.viewport.isDirty=!0,!1):(L.emit(k.SCULPT_START,{position:s.position,radius:s.radius,operation:this.sculptingSystem.brushManager.state.operation,modifiers:e.modifiers}),this.sculptingSystem.performSculpting(),this.viewport.isDirty=!0,!1)}}handleCameraRotate(e){this.isSculpting||(this.isNavigating=!0)}handleSculptEnd(){this.sculptingSystem.brushManager.state.operation===O.MOVE&&this.sculptingSystem.brushManager.endMove(),this.sculptingSystem.editingCache&&this.sculptingSystem.editingCache.isActive&&this.sculptingSystem.syncEditingCache(),this.isSculpting&&L.emit(k.SCULPT_END),this.isSculpting=!1,this.isNavigating=!1}handleSculptHover(e){if(this.mouseThrottleInterval>0){const a=performance.now();if(a-this.lastMouseMoveTime<this.mouseThrottleInterval)return;this.lastMouseMoveTime=a}const t=this.sculptingSystem.state.getBrushSettings();if(this.isNavigating&&e.mouse.button!==-1){this.sculptingSystem.showBrushPreview&&(this.sculptingSystem.showBrushPreview=!1,this.viewport.hideBrushPreview());return}if(this.isSculpting&&!this.isNavigating&&this.sculptingSystem.brushManager.state.operation===O.MOVE){this.handleMoveDrag(e);return}const i=this.input.getNormalizedMousePosition(),s=this.viewport.getRayFromViewport(i.x,i.y);s&&(this.sculptingSystem.lastRayOrigin=s.origin,this.sculptingSystem.lastRayDirection=s.direction,this.sculptingSystem.raycastHandler.performRaycast(s.origin,s.direction,!0).then(a=>{if(this.isNavigating&&this.input.mouse.button!==-1){this.sculptingSystem.showBrushPreview&&(this.sculptingSystem.showBrushPreview=!1,this.viewport.hideBrushPreview());return}if(a&&a.position){if(this.sculptingSystem.lastRaycastHit=a,this.sculptingSystem.showBrushPreview=!0,this.viewport.updateBrushPreview(a.position,this.sculptingSystem.state.getBrushSettings().radius,a.normal,a.distance),this.isSculpting&&!this.isNavigating&&this.sculptingSystem.brushManager.state.operation!==O.MOVE){t.position.set(a.position),t.normal.set(a.normal);const n=Date.now();n-this.lastSculptTime>=this.sculptThrottleInterval&&(this.sculptingSystem.performSculpting(),this.lastSculptTime=n)}}else this.sculptingSystem.lastRaycastHit=null,this.sculptingSystem.showBrushPreview=!1,this.viewport.hideBrushPreview()}))}handleMoveDrag(e){const t=[e.mouse.x,e.mouse.y],i=this.sculptingSystem.lastRaycastHit?this.sculptingSystem.lastRaycastHit.distance:10;this.sculptingSystem.brushManager.updateMove(t,this.viewport.camera.viewMatrix,this.viewport.camera.projectionMatrix,this.screenDeltaToWorldDelta.bind(this),i,this.viewport.camera,this.viewport.canvas);const s=this.sculptingSystem.state.getBrushSettings();this.sculptingSystem.showBrushPreview=!0,this.viewport.updateBrushPreview(s.position,s.radius,s.normal,i),this.sculptingSystem.performSculpting(),this.viewport.isDirty=!0}screenDeltaToWorldDelta(e,t,i,s){if(!i||!i.viewMatrix||!s||!t||t<=0)return[0,0,0];const a=i.viewMatrix;if(!a||a.length<16)return[0,0,0];const n=[a[0],a[4],a[8]],o=[a[1],a[5],a[9]],c=1/i.projectionMatrix[5],p=2*t*c,u=window.devicePixelRatio||1,g=s.width/u,v=s.height/u,w=s.width/s.height,y=p*w,_=p/v,f=y/g,m=n[0]*e.x*f+o[0]*-e.y*_,S=n[1]*e.x*f+o[1]*-e.y*_,x=n[2]*e.x*f+o[2]*-e.y*_;return isNaN(m)||isNaN(S)||isNaN(x)?[0,0,0]:[m,S,x]}handleKeyDown(e){this.modifierKeys.ctrl=e.ctrlKey,this.modifierKeys.shift=e.shiftKey,this.modifierKeys.alt=e.altKey,this.modifierKeys.meta=e.metaKey,e.key==="Control"||e.key,e.key==="Shift"&&!this.isShiftBrushActive&&(this.isShiftBrushActive=!0,this.previousBrushType=this.sculptingSystem.state.getBrushSettings().type,this.sculptingSystem.setBrushType(O.ERODE))}handleKeyUp(e){this.modifierKeys.ctrl=e.ctrlKey,this.modifierKeys.shift=e.shiftKey,this.modifierKeys.alt=e.altKey,this.modifierKeys.meta=e.metaKey,e.key==="Control"||e.key,e.key==="Shift"&&this.isShiftBrushActive&&(this.isShiftBrushActive=!1,this.previousBrushType!==null&&(this.sculptingSystem.setBrushType(this.previousBrushType),this.previousBrushType=null))}shouldHandleMouseEvent(e){if(e.button!==0)return!1;const t=this.viewport.canvas.getBoundingClientRect(),i=(e.clientX-t.left)/t.width*2-1,s=-((e.clientY-t.top)/t.height)*2+1,a=d.sculpting.input.borderThresholdPercent,n=Math.abs(i),o=Math.abs(s);return n>1-a*2||o>1-a*2?!1:this.sculptingSystem.showBrushPreview}getState(){return{isSculpting:this.isSculpting,isNavigating:this.isNavigating,modifierKeys:this.modifierKeys,tabletPressure:this.tabletPressure,pointerType:this.pointerType}}setMouseThrottleFPS(e){this.mouseThrottleFPS=e,this.mouseThrottleInterval=e>0?1e3/e:0}}const D={COMPUTE:GPUShaderStage.COMPUTE,FRAGMENT:GPUShaderStage.FRAGMENT,VERTEX:GPUShaderStage.VERTEX,COMPUTE_FRAGMENT:GPUShaderStage.COMPUTE|GPUShaderStage.FRAGMENT,VERTEX_FRAGMENT:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,ALL:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT|GPUShaderStage.COMPUTE},Qe={cache:{entries:[{binding:0,visibility:D.COMPUTE_FRAGMENT,buffer:{type:"storage"}},{binding:1,visibility:D.COMPUTE_FRAGMENT,buffer:{type:"uniform"}}]},cacheReadOnly:{entries:[{binding:0,visibility:D.COMPUTE_FRAGMENT,buffer:{type:"read-only-storage"}},{binding:1,visibility:D.COMPUTE_FRAGMENT,buffer:{type:"uniform"}}]},voxelHashMap:{entries:[{binding:0,visibility:D.COMPUTE_FRAGMENT,buffer:{type:"storage"}},{binding:1,visibility:D.COMPUTE_FRAGMENT,buffer:{type:"storage"}},{binding:2,visibility:D.COMPUTE_FRAGMENT,buffer:{type:"uniform"}},{binding:3,visibility:D.COMPUTE_FRAGMENT,buffer:{type:"storage"}},{binding:4,visibility:D.COMPUTE_FRAGMENT,buffer:{type:"storage"}}]},voxelHashMapReadOnly:{entries:[{binding:0,visibility:D.COMPUTE_FRAGMENT,buffer:{type:"read-only-storage"}},{binding:1,visibility:D.COMPUTE_FRAGMENT,buffer:{type:"read-only-storage"}},{binding:2,visibility:D.COMPUTE_FRAGMENT,buffer:{type:"uniform"}},{binding:3,visibility:D.COMPUTE_FRAGMENT,buffer:{type:"read-only-storage"}},{binding:4,visibility:D.COMPUTE_FRAGMENT,buffer:{type:"read-only-storage"}}]},uniform:{entries:[{binding:0,visibility:D.ALL,buffer:{type:"uniform"}}]},storage:{entries:[{binding:0,visibility:D.COMPUTE,buffer:{type:"storage"}}]},textureSampling:{entries:[{binding:0,visibility:D.FRAGMENT,texture:{sampleType:"float"}},{binding:1,visibility:D.FRAGMENT,sampler:{}}]}};class et{constructor(e){this.device=e,this.layoutCache=new Map}getStandardLayout(e,t){const i=`${e}_${t||""}`;if(this.layoutCache.has(i))return this.layoutCache.get(i);const s=Qe[e];if(!s)throw new Error(`Unknown standard layout: ${e}`);const a=this.device.createBindGroupLayout({label:t||`standardLayout.${e}`,entries:s.entries});return this.layoutCache.set(i,a),a}createLayout(e,t){return this.device.createBindGroupLayout({label:t,entries:e})}getFromPipeline(e,t){return e.getBindGroupLayout(t)}}class J{constructor(e){this.device=e,this.entries=[]}addBuffer(e,t){if(!t)throw new Error(`Buffer at binding ${e} is null or undefined`);return this.entries.push({binding:e,resource:{buffer:t}}),this}addTexture(e,t){if(!t)throw new Error(`Texture at binding ${e} is null or undefined`);return this.entries.push({binding:e,resource:t.createView()}),this}addTextureView(e,t){if(!t)throw new Error(`Texture view at binding ${e} is null or undefined`);return this.entries.push({binding:e,resource:t}),this}addSampler(e,t){if(!t)throw new Error(`Sampler at binding ${e} is null or undefined`);return this.entries.push({binding:e,resource:t}),this}build(e,t){return this.device.createBindGroup({label:t,layout:e,entries:this.entries})}}class tt{constructor(e){this.device=e,this.layoutFactory=new et(e)}createCacheBindGroup(e,t,i=!1){const s=i?"cacheReadOnly":"cache",a=this.layoutFactory.getStandardLayout(s);return new J(this.device).addBuffer(0,e).addBuffer(1,t).build(a,`cache.${i?"readOnly":"write"}`)}createVoxelHashMapBindGroup(e,t=!1){const i=t?"voxelHashMapReadOnly":"voxelHashMap",s=this.layoutFactory.getStandardLayout(i);return new J(this.device).addBuffer(0,e.hashTable).addBuffer(1,e.voxelData).addBuffer(2,e.params).addBuffer(3,e.occupancyHash).addBuffer(4,e.occupancyData).build(s,`voxelHashMap.${t?"readOnly":"write"}`)}createUniformBindGroup(e,t="uniform"){const i=this.layoutFactory.getStandardLayout("uniform");return new J(this.device).addBuffer(0,e).build(i,t)}createStorageBindGroup(e,t="storage"){const i=this.layoutFactory.getStandardLayout("storage");return new J(this.device).addBuffer(0,e).build(i,t)}createTextureSamplingBindGroup(e,t,i="textureSampling"){const s=this.layoutFactory.getStandardLayout("textureSampling");return new J(this.device).addTexture(0,e).addSampler(1,t).build(s,i)}getLayoutFactory(){return this.layoutFactory}createBuilder(){return new J(this.device)}}let we=null;function ai(r){return we=new tt(r),we}function Re(){if(!we)throw new Error("BindGroupFactory not initialized. Call initializeBindGroupFactory(device) first.");return we}const ri=Object.freeze(Object.defineProperty({__proto__:null,BindGroupBuilder:J,BindGroupFactory:tt,BindGroupLayoutFactory:et,ShaderStages:D,StandardLayouts:Qe,getBindGroupFactory:Re,initializeBindGroupFactory:ai},Symbol.toStringTag,{value:"Module"})),W=globalThis.GPUBufferUsage,Te=globalThis.GPUShaderStage,ni=globalThis.GPUMapMode;class oi{constructor(e,t,i,s){this.gpu=e,this.dimensions=t,this.voxelSize=i,this.origin=Z(s),this.voxelCount=t.x*t.y*t.z,this.createBuffers(),this.createBindGroups(),this.isDirty=!1,this.isAllocated=!0,this.dirtyBounds={min:G(1/0,1/0,1/0),max:G(-1/0,-1/0,-1/0),isValid:!1}}createBuffers(){const e=this.gpu.device;this.dataBuffer=e.createBuffer({label:"AdaptiveCache.dataBuffer",size:this.voxelCount*d.bufferSizes.CACHE_ELEMENT_SIZE,usage:W.STORAGE|W.COPY_SRC|W.COPY_DST}),this.snapshotBuffer=e.createBuffer({label:"AdaptiveCache.snapshotBuffer",size:this.voxelCount*d.bufferSizes.CACHE_ELEMENT_SIZE,usage:W.STORAGE|W.COPY_DST}),this.metadataBuffer=e.createBuffer({label:"AdaptiveCache.metadataBuffer",size:d.bufferSizes.CACHE_METADATA_SIZE,usage:W.UNIFORM|W.COPY_DST}),this.updateMetadata()}ensureSnapshotBuffer(){}createBindGroups(){const e=Re(),t=e.getLayoutFactory();this.dataBindGroupLayout=t.getStandardLayout("cache","AdaptiveCache.dataBindGroupLayout"),this.dataBindGroupLayoutReadOnly=t.getStandardLayout("cacheReadOnly","AdaptiveCache.dataBindGroupLayoutReadOnly"),this.dataBindGroup=e.createCacheBindGroup(this.dataBuffer,this.metadataBuffer,!1),this.dataBindGroupReadOnly=e.createCacheBindGroup(this.dataBuffer,this.metadataBuffer,!0),this.grabBindGroupLayout=this.gpu.device.createBindGroupLayout({label:"AdaptiveCache.grabBindGroupLayout",entries:[{binding:0,visibility:Te.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:Te.COMPUTE,buffer:{type:"uniform"}},{binding:2,visibility:Te.COMPUTE,buffer:{type:"read-only-storage"}}]}),this.grabBindGroup=this.gpu.device.createBindGroup({label:"AdaptiveCache.grabBindGroup",layout:this.grabBindGroupLayout,entries:[{binding:0,resource:{buffer:this.dataBuffer}},{binding:1,resource:{buffer:this.metadataBuffer}},{binding:2,resource:{buffer:this.snapshotBuffer}}]})}ensureGrabBindGroup(){return this.grabBindGroup}createRaycastCompatibleBindGroup(e){const t=Re();return this.raycastBindGroup=t.createBuilder().addBuffer(0,this.dataBuffer).addBuffer(1,this.metadataBuffer).build(e,"AdaptiveCache.raycastBindGroup"),this.raycastBindGroup}captureSnapshot(e){e.copyBufferToBuffer(this.dataBuffer,0,this.snapshotBuffer,0,this.voxelCount*d.bufferSizes.CACHE_ELEMENT_SIZE)}updateMetadata(){const e=new ArrayBuffer(d.bufferSizes.CACHE_METADATA_SIZE),t=new Float32Array(e),i=new Uint32Array(e);t.fill(0),t[0]=this.origin[0],t[1]=this.origin[1],t[2]=this.origin[2],t[3]=this.voxelSize,i[4]=this.dimensions.x,i[5]=this.dimensions.y,i[6]=this.dimensions.z,i[7]=1,this.gpu.device.queue.writeBuffer(this.metadataBuffer,0,e)}worldToIndex(e){const t=M();be(t,e,this.origin);const i=G(Math.floor(t[0]/this.voxelSize),Math.floor(t[1]/this.voxelSize),Math.floor(t[2]/this.voxelSize));return i[0]<0||i[0]>=this.dimensions.x||i[1]<0||i[1]>=this.dimensions.y||i[2]<0||i[2]>=this.dimensions.z?-1:i[0]+i[1]*this.dimensions.x+i[2]*this.dimensions.x*this.dimensions.y}voxelToWorld(e){const t=M();return t[0]=this.origin[0]+(e[0]+.5)*this.voxelSize,t[1]=this.origin[1]+(e[1]+.5)*this.voxelSize,t[2]=this.origin[2]+(e[2]+.5)*this.voxelSize,t}containsPosition(e){return this.worldToIndex(e)>=0}getWorldBounds(){const e=G(this.origin[0],this.origin[1],this.origin[2]),t=G(this.origin[0]+this.dimensions.x*this.voxelSize,this.origin[1]+this.dimensions.y*this.voxelSize,this.origin[2]+this.dimensions.z*this.voxelSize);return{min:e,max:t}}markDirty(){this.isDirty=!0}clearDirty(){this.isDirty=!1,this.resetDirtyBounds()}updateDirtyBounds(e,t){this.dirtyBounds.isValid?(je(this.dirtyBounds.min,this.dirtyBounds.min,e),Ke(this.dirtyBounds.max,this.dirtyBounds.max,t)):(ce(this.dirtyBounds.min,e),ce(this.dirtyBounds.max,t),this.dirtyBounds.isValid=!0)}resetDirtyBounds(){this.dirtyBounds.min[0]=1/0,this.dirtyBounds.min[1]=1/0,this.dirtyBounds.min[2]=1/0,this.dirtyBounds.max[0]=-1/0,this.dirtyBounds.max[1]=-1/0,this.dirtyBounds.max[2]=-1/0,this.dirtyBounds.isValid=!1}getDirtyRegionDispatch(){if(!this.dirtyBounds.isValid)return{offset:G(0,0,0),dimensions:Z(this.dimensions),workgroups:G(Math.ceil(this.dimensions.x/8),Math.ceil(this.dimensions.y/8),Math.ceil(this.dimensions.z/4))};const e=G(Math.max(0,Math.floor(this.dirtyBounds.min[0])),Math.max(0,Math.floor(this.dirtyBounds.min[1])),Math.max(0,Math.floor(this.dirtyBounds.min[2]))),t=G(Math.min(this.dimensions.x,Math.ceil(this.dirtyBounds.max[0]+1)),Math.min(this.dimensions.y,Math.ceil(this.dirtyBounds.max[1]+1)),Math.min(this.dimensions.z,Math.ceil(this.dirtyBounds.max[2]+1))),i=G(t[0]-e[0],t[1]-e[1],t[2]-e[2]),s=G(Math.ceil(i[0]/8),Math.ceil(i[1]/8),Math.ceil(i[2]/4));return{offset:e,dimensions:i,workgroups:s}}destroy(){this.dataBuffer&&(this.dataBuffer.destroy(),this.dataBuffer=null),this.snapshotBuffer&&(this.snapshotBuffer.destroy(),this.snapshotBuffer=null),this.metadataBuffer&&(this.metadataBuffer.destroy(),this.metadataBuffer=null),this.isAllocated=!1}async fillFromVoxels(e,t){var u,g;if(!e||!t)throw new Error("Valid voxel system and fill pipeline required");const i=this.gpu.device.createCommandEncoder({label:"AdaptiveCache.fillFromVoxels"}),s=i.beginComputePass({label:"AdaptiveCache.fillFromVoxels.computePass"});s.setPipeline(t);let a;if(e.bindGroups&&e.bindGroups.read)a=e.bindGroups.read;else if(e.readBindGroup)a=e.readBindGroup;else if(e.createFullReadBindGroup){const v=(u=this.gpu.bindGroupLayouts)==null?void 0:u.voxelRead;v&&(a=e.createFullReadBindGroup(v))}else if(e.createBindGroup){const v=(g=this.gpu.bindGroupLayouts)==null?void 0:g.voxelRead;v&&(a=e.createBindGroup(v))}if(!a)throw new Error("Could not get voxel system bind group for reading");s.setBindGroup(0,a),s.setBindGroup(1,this.dataBindGroup);const n=Math.ceil(this.dimensions.x/8),o=Math.ceil(this.dimensions.y/8),l=Math.ceil(this.dimensions.z/4),c=65535;if(!(n>c||o>c||l>c))s.dispatchWorkgroups(n,o,l);else throw new Error("Cache dimensions exceed WebGPU limits, chunking not implemented yet");return s.end(),this.gpu.device.queue.submit([i.finish()]),this.clearDirty(),!0}async syncToVoxels(e,t){if(!t)throw new Error("Valid sync pipeline required");return t.syncToVoxels(this,e)}async applyBrush(e,t){if(!t)throw new Error("Valid brush pipeline required");return t.applyBrush(this,e)}async readback(){const e=this.voxelCount*d.bufferSizes.CACHE_ELEMENT_SIZE,t=this.gpu.device.createBuffer({size:e,usage:W.COPY_DST|W.MAP_READ}),i=this.gpu.device.createCommandEncoder();i.copyBufferToBuffer(this.dataBuffer,0,t,0,e),this.gpu.device.queue.submit([i.finish()]),await t.mapAsync(ni.READ);const s=new Float32Array(t.getMappedRange()),a=new Float32Array(s);return t.unmap(),t.destroy(),a}}class li{constructor(e){if(!e)throw new Error("CacheAllocator requires config");if(!e.maxCacheSize)throw new Error("CacheAllocator requires config.maxCacheSize");if(!e.baseVoxelSize)throw new Error("CacheAllocator requires config.baseVoxelSize");this.maxCacheSize=e.maxCacheSize,this.baseVoxelSize=e.baseVoxelSize,this.maxSculptingDistance=Ge.adaptiveCache.maxSculptingDistance,this.MIN_DEPTH_RANGE_FACTOR=1,this.BRUSH_DEPTH_FACTOR=10}calculateBounds(e,t,i){const s=M(),a=M(),n=M();Ee(n,e,t);for(let o=0;o<3;o++)s[o]=Math.min(e[o],n[o])-i,a[o]=Math.max(e[o],n[o])+i;return{min:s,max:a}}calculateAdaptiveVoxelSize(e){return this.baseVoxelSize}alignToVoxelGrid(e,t){const i={min:M(),max:M()};return xe(i.min,Math.floor(e.min[0]/t)*t,Math.floor(e.min[1]/t)*t,Math.floor(e.min[2]/t)*t),xe(i.max,Math.ceil(e.max[0]/t)*t,Math.ceil(e.max[1]/t)*t,Math.ceil(e.max[2]/t)*t),i}calculateCacheDimensions(e,t,i=0){const s=M();ee(s,e.max,e.min);const a={x:Math.round(s[0]/t),y:Math.round(s[1]/t),z:Math.round(s[2]/t)};a.x=Math.max(1,a.x),a.y=Math.max(1,a.y),a.z=Math.max(1,a.z);let n=this.maxCacheSize;if(i>0)if(i>8)n=32;else{const c=i/8;n=Math.round(this.maxCacheSize*(1-c*.875)),n=Math.max(32,n)}a.x=Math.min(a.x,n),a.y=Math.min(a.y,n),a.z=Math.min(a.z,n);const o=16*1024*1024,l=a.x*a.y*a.z;if(l>o){const c=Math.cbrt(o/l);a.x=Math.max(1,Math.floor(a.x*c)),a.y=Math.max(1,Math.floor(a.y*c)),a.z=Math.max(1,Math.floor(a.z*c))}return a}allocateForMove(e,t,i,s){const a=M();ee(a,t,e);const n=this.calculateBounds(e,a,i),o=this.calculateAdaptiveVoxelSize(s),l=this.alignToVoxelGrid(n,o),c=this.calculateCacheDimensions(l,o,s);return{worldBounds:l,voxelSize:o,dimensions:c,totalVoxels:c.x*c.y*c.z}}allocateForAccumulating(e,t,i){const s={min:M(),max:M()},a=G(t,t,t);ee(s.min,e,a),Ee(s.max,e,a);const n=this.calculateAdaptiveVoxelSize(i),o=this.alignToVoxelGrid(s,n),l=this.calculateCacheDimensions(o,n,i);return{worldBounds:o,voxelSize:n,dimensions:l,totalVoxels:l.x*l.y*l.z}}allocateScreenAligned(e,t,i,s){const{position:a,fov:n,near:o,far:l,viewMatrix:c}=t,{width:p,height:u}=i,g=G(c[0],c[4],c[8]),v=G(c[1],c[5],c[9]),w=G(-c[2],-c[6],-c[10]),y=M();ee(y,e,a);const _=ve(y,w),f=Math.min(1,_/this.maxSculptingDistance),m=8;let S={min:M(),max:M()};if(_>m){const E=G(s*1.5,s*1.5,s*1.5);ee(S.min,e,E),Ee(S.max,e,E)}else{const E=2*_*Math.tan(n/2),C=p/u,z=E*C,F=Math.max(.05,1-f*.95),B=Math.max(.1,1-f*.8),P=E*B,V=z*B,H=s*.25*F,I=(s+H)/_*F,st=P+2*I*P,at=V+2*I*V,rt=Math.sqrt(V*V+P*P),Le=Math.max(s*this.BRUSH_DEPTH_FACTOR,rt*this.MIN_DEPTH_RANGE_FACTOR*.5)*F/2;S={min:G(1/0,1/0,1/0),max:G(-1/0,-1/0,-1/0)};const nt=M();ne(nt,a,w,_);const ot=[_-Le,_+Le];for(const Oe of ot){const ke=Oe/_,lt=st/2*ke,ct=at/2*ke,Ve=M();ne(Ve,a,w,Oe);for(let Be=-1;Be<=1;Be+=2)for(let Me=-1;Me<=1;Me+=2){const j=M();ce(j,Ve),ne(j,j,g,Be*ct),ne(j,j,v,Me*lt),je(S.min,S.min,j),Ke(S.max,S.max,j)}}}const x=this.calculateAdaptiveVoxelSize(_),T=this.alignToVoxelGrid(S,x),A=this.calculateCacheDimensions(T,x,_);return{worldBounds:T,voxelSize:x,dimensions:A,totalVoxels:A.x*A.y*A.z,surfaceDepth:_,coverage:{width:S.max[0]-S.min[0],height:S.max[1]-S.min[1],depth:_>m?s*3:S.max[2]-S.min[2]}}}}class it{constructor(e){this.allocator=e}}class ci extends it{constructor(e){super(e),this.originalSurfacePoint=null,this.surfaceNormal=null,this.initialDepth=null,this.isActive=!1}onMouseDown(e,t,i,s,a){return this.originalSurfacePoint=Z(e),this.surfaceNormal=Z(t),this.initialDepth=i,this.isActive=!0,this.brushRadius=s,this.cameraDistance=a,this.allocator.allocateForMove(e,e,s,a)}onMouseMove(e,t,i){return this.isActive?(this.brushRadius=t,this.cameraDistance=i,this.allocator.allocateForMove(this.originalSurfacePoint,e,t,i)):null}onMouseUp(){const e=this.isActive;return this.isActive=!1,this.originalSurfacePoint=null,this.surfaceNormal=null,this.initialDepth=null,e}getDisplacement(e){if(!this.originalSurfacePoint)return M();const t=M();return ee(t,e,this.originalSurfacePoint),t}}class qe extends it{constructor(e,t=5){super(e),this.commitInterval=t,this.framesSinceCommit=0,this.lastCommitPosition=null,this.minCommitDistance=2}onMouseDown(e,t,i){return this.lastCommitPosition=Z(e),this.framesSinceCommit=0,this.allocator.allocateForAccumulating(e,t,i)}onMouseMove(e,t,i){this.framesSinceCommit++;const s=Je(e,this.lastCommitPosition);return this.framesSinceCommit>=this.commitInterval||s>=this.minCommitDistance?(this.lastCommitPosition=Z(e),this.framesSinceCommit=0,{allocation:this.allocator.allocateForAccumulating(e,t,i),shouldCommit:!0}):{allocation:null,shouldCommit:!1}}onMouseUp(){const e=this.framesSinceCommit>0;return this.lastCommitPosition=null,this.framesSinceCommit=0,e}}const ui=`
${U.metadata}
${U.constants}
${Pe}

// CENTER-BASED SYSTEM: Simplified cache fill using trilinear interpolation
// No longer need complex extrapolation since we use interpolation directly

// Voxel hash map bindings (read-only)
@group(0) @binding(0) var<storage, read> hash_table: array<vec4<i32>>;
@group(0) @binding(1) var<storage, read> voxel_data: array<f32>;
@group(0) @binding(2) var<uniform> voxel_params: VoxelHashMapParams;
@group(0) @binding(3) var<storage, read> occupancy_hash: array<vec4<i32>>;
@group(0) @binding(4) var<storage, read> occupancy_data: array<u32>;

// Cache bindings (write)
@group(1) @binding(0) var<storage, read_write> cache_data: array<f32>;
@group(1) @binding(1) var<uniform> cache_metadata: CacheMetadata;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = vec3<u32>(cache_metadata.dimensions);
    
    // Check bounds
    if (global_id.x >= dims.x || global_id.y >= dims.y || global_id.z >= dims.z) {
        return;
    }
    
    // ROBUST COORDINATE SYSTEM: Cache origin is corner, but sample at voxel centers
    // cache[i] stores SDF sampled at: origin + (i + 0.5) * voxel_size
    let cache_voxel_center = vec3<f32>(global_id) + 0.5;
    let world_pos = cache_metadata.origin + cache_voxel_center * cache_metadata.voxel_size;
    
    // SIMPLIFIED APPROACH: Always query the voxel hash map
    // The hash map will return appropriate values for empty regions
    let sdf_value = query_voxel_sdf_trilinear(world_pos, voxel_params.voxel_size, &hash_table, &voxel_data);
    
    // CENTER-BASED SYSTEM: Always use trilinear interpolation result
    // The query_voxel_sdf_trilinear function handles all interpolation and fallback cases
    
    // Clamp the initial value to prevent starting with extreme values
    // This helps prevent explosion during brush operations
    let clamped_sdf = clamp(sdf_value, -cache_metadata.voxel_size * 10.0, cache_metadata.voxel_size * 10.0);
    
    // Calculate linear index in cache
    let cache_idx = global_id.x + 
                    global_id.y * dims.x + 
                    global_id.z * dims.x * dims.y;
    
    // Write to cache
    cache_data[cache_idx] = clamped_sdf;
}
`,pe=globalThis.GPUShaderStage;class di{constructor(e){this.gpu=e,this.pipeline=null,this.bindGroupLayout=null}async initialize(){const e=this.gpu.device.createShaderModule({label:"CacheFillPipeline.shaderModule",code:ui});this.cacheBindGroupLayout=this.gpu.device.createBindGroupLayout({label:"CacheFillPipeline.cacheBindGroupLayout",entries:[{binding:0,visibility:pe.COMPUTE|pe.FRAGMENT,buffer:{type:"storage"}},{binding:1,visibility:pe.COMPUTE|pe.FRAGMENT,buffer:{type:"uniform"}}]});const t=this.gpu.device.createPipelineLayout({label:"CacheFillPipeline.pipelineLayout",bindGroupLayouts:[this.gpu.bindGroupLayouts.voxelRead,this.cacheBindGroupLayout]});return this.pipeline=this.gpu.device.createComputePipeline({label:"CacheFillPipeline",layout:t,compute:{module:e,entryPoint:"main"}}),this.pipeline}getPipeline(){if(!this.pipeline)throw new Error("Pipeline not initialized. Call initialize() first.");return this.pipeline}destroy(){this.pipeline&&(this.pipeline.destroy(),this.pipeline=null),this.cacheBindGroupLayout&&(this.cacheBindGroupLayout.destroy(),this.cacheBindGroupLayout=null),this.gpu=null}}const hi=`
${U.metadata}
${U.constants}
${U.utilities}

// Empty space SDF value (matches voxelHashMap.wgsl.js)
const VOXEL_EMPTY_SDF: f32 = 10.0;

// Maximum reasonable SDF value (in voxel units)
const MAX_SDF_VALUE: f32 = 50.0;

// Empty space threshold multiplier for consistency
const EMPTY_THRESHOLD_MULTIPLIER: f32 = 50.0;

// Brush operation types
const BRUSH_MOVE: u32 = 0u;
const BRUSH_BUMP: u32 = 1u;
const BRUSH_SMOOTH: u32 = 2u;

// Falloff types
const FALLOFF_CONSTANT: u32 = 0u;
const FALLOFF_LINEAR: u32 = 1u;
const FALLOFF_SMOOTH: u32 = 2u;
const FALLOFF_GAUSSIAN: u32 = 3u;
const FALLOFF_SHARP: u32 = 4u;

struct BrushParams {
    position: vec3<f32>,      // Brush center in world space
    radius: f32,              // Brush radius in world units
    strength: f32,            // Brush strength [0-1]
    operation: u32,           // Brush operation type
    falloff_type: u32,        // Falloff function type
    target_value: f32,        // For smooth operations
    normal: vec3<f32>,        // Surface normal (unused for MOVE)
    smooth_bias: f32,         // Smooth brush strength bias
    grab_original_pos: vec3<f32>, // For MOVE: where the grab started
    _padding2: u32,           // Align to 16 bytes
}

// Brush parameters (uniform)
@group(0) @binding(0) var<uniform> brush: BrushParams;

// Cache data (read-write)
@group(1) @binding(0) var<storage, read_write> cache_data: array<f32>;
@group(1) @binding(1) var<uniform> cache_metadata: CacheMetadata;

// Calculate brush influence at a given distance
fn calculate_falloff(distance: f32, radius: f32, falloff_type: u32) -> f32 {
    let normalized_dist = saturate(distance / radius);
    
    switch (falloff_type) {
        case FALLOFF_CONSTANT: {
            return select(0.0, 1.0, distance <= radius);
        }
        case FALLOFF_LINEAR: {
            return 1.0 - normalized_dist;
        }
        case FALLOFF_SMOOTH: {
            // Smoother falloff that better matches the visual radius
            let t = 1.0 - normalized_dist;
            return t * t * (3.0 - 2.0 * t); // Standard smoothstep
        }
        case FALLOFF_GAUSSIAN: {
            // Gaussian falloff
            let sigma = 0.5; // Controls sharpness
            let t = normalized_dist / sigma;
            return exp(-0.5 * t * t);
        }
        case FALLOFF_SHARP: {
            // Sharp falloff near edges
            let t = 1.0 - normalized_dist;
            return t * t * t * t;
        }
        default: {
            return 0.0;
        }
    }
}

// Sample SDF from cache at any world position using trilinear interpolation
fn sample_cache_sdf(world_pos: vec3<f32>) -> f32 {
    // Convert world position to cache local position
    let local_pos = world_pos - cache_metadata.origin;
    
    // Find which cache voxel contains this position
    let voxel_pos = (local_pos / cache_metadata.voxel_size) - 0.5;
    
    // Clamp to valid bounds
    let clamped_voxel_pos = clamp(voxel_pos, vec3<f32>(0.0), vec3<f32>(cache_metadata.dimensions) - vec3<f32>(1.001));
    
    // Trilinear interpolation using clamped position
    let voxel_coord = floor(clamped_voxel_pos);
    let fract_pos = fract(clamped_voxel_pos);
    
    // Get integer coordinates
    let x0 = u32(voxel_coord.x);
    let y0 = u32(voxel_coord.y);
    let z0 = u32(voxel_coord.z);
    
    let x1 = min(x0 + 1u, cache_metadata.dimensions.x - 1u);
    let y1 = min(y0 + 1u, cache_metadata.dimensions.y - 1u);
    let z1 = min(z0 + 1u, cache_metadata.dimensions.z - 1u);
    
    // Sample 8 corners from cache_data
    let v000 = cache_data[x0 + y0 * cache_metadata.dimensions.x + z0 * cache_metadata.dimensions.x * cache_metadata.dimensions.y];
    let v100 = cache_data[x1 + y0 * cache_metadata.dimensions.x + z0 * cache_metadata.dimensions.x * cache_metadata.dimensions.y];
    let v010 = cache_data[x0 + y1 * cache_metadata.dimensions.x + z0 * cache_metadata.dimensions.x * cache_metadata.dimensions.y];
    let v110 = cache_data[x1 + y1 * cache_metadata.dimensions.x + z0 * cache_metadata.dimensions.x * cache_metadata.dimensions.y];
    let v001 = cache_data[x0 + y0 * cache_metadata.dimensions.x + z1 * cache_metadata.dimensions.x * cache_metadata.dimensions.y];
    let v101 = cache_data[x1 + y0 * cache_metadata.dimensions.x + z1 * cache_metadata.dimensions.x * cache_metadata.dimensions.y];
    let v011 = cache_data[x0 + y1 * cache_metadata.dimensions.x + z1 * cache_metadata.dimensions.x * cache_metadata.dimensions.y];
    let v111 = cache_data[x1 + y1 * cache_metadata.dimensions.x + z1 * cache_metadata.dimensions.x * cache_metadata.dimensions.y];
    
    // Trilinear interpolation
    let x00 = mix(v000, v100, fract_pos.x);
    let x10 = mix(v010, v110, fract_pos.x);
    let x01 = mix(v001, v101, fract_pos.x);
    let x11 = mix(v011, v111, fract_pos.x);
    
    let y_0 = mix(x00, x10, fract_pos.y);
    let y_1 = mix(x01, x11, fract_pos.y);
    
    return mix(y_0, y_1, fract_pos.z);
}



// Smooth minimum for SDF union
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

// Smooth maximum for SDF subtraction
fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    return -smooth_min(-a, -b, k);
}

// Apply SMOOTH brush - morphological operations (erode/dilate)
fn apply_smooth_brush(
    current_sdf: f32, 
    influence: f32,
    cache_idx: u32,
    dims: vec3<u32>,
    world_pos: vec3<f32>,
    global_id: vec3<u32>
) -> f32 {
    // SMOOTH brush: Morphological operations for proper SDF smoothing
    
    // Early exit for weak influence
    if (influence < 0.01) {
        return current_sdf;
    }
    
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    // Pre-calculate strides for faster indexing
    let stride_x = 1u;
    let stride_y = dims.x;
    let stride_z = dims.x * dims.y;
    
    // Morphological operations: min for dilation, max for erosion
    var min_sdf = current_sdf;
    var max_sdf = current_sdf;
    
    // Sample radius based on influence (1-3 voxels)
    let sample_radius = u32(ceil(influence * 2.0));
    
    // Efficient sampling pattern
    // For radius 1: 6 neighbors (faces)
    // For radius 2: add 12 edges  
    // For radius 3: add 8 corners
    
    // Face neighbors (always sample)
    if (x > 0u) {
        let val = cache_data[cache_idx - stride_x];
        min_sdf = min(min_sdf, val);
        max_sdf = max(max_sdf, val);
    }
    if (x < dims.x - 1u) {
        let val = cache_data[cache_idx + stride_x];
        min_sdf = min(min_sdf, val);
        max_sdf = max(max_sdf, val);
    }
    if (y > 0u) {
        let val = cache_data[cache_idx - stride_y];
        min_sdf = min(min_sdf, val);
        max_sdf = max(max_sdf, val);
    }
    if (y < dims.y - 1u) {
        let val = cache_data[cache_idx + stride_y];
        min_sdf = min(min_sdf, val);
        max_sdf = max(max_sdf, val);
    }
    if (z > 0u) {
        let val = cache_data[cache_idx - stride_z];
        min_sdf = min(min_sdf, val);
        max_sdf = max(max_sdf, val);
    }
    if (z < dims.z - 1u) {
        let val = cache_data[cache_idx + stride_z];
        min_sdf = min(min_sdf, val);
        max_sdf = max(max_sdf, val);
    }
    
    // Edge neighbors for stronger smoothing
    if (sample_radius >= 2u && influence > 0.4) {
        // XY edges
        if (x > 0u && y > 0u) {
            let val = cache_data[cache_idx - stride_x - stride_y];
            min_sdf = min(min_sdf, val);
            max_sdf = max(max_sdf, val);
        }
        if (x < dims.x - 1u && y < dims.y - 1u) {
            let val = cache_data[cache_idx + stride_x + stride_y];
            min_sdf = min(min_sdf, val);
            max_sdf = max(max_sdf, val);
        }
    }
    
    // Combine erosion and dilation based on surface proximity
    // Near surface (small SDF): use average of min/max for smoothing
    // Far from surface: use weighted blend favoring current value
    let surface_proximity = saturate(1.0 - abs(current_sdf) / (cache_metadata.voxel_size * 5.0));
    let morphed_sdf = mix(current_sdf, (min_sdf + max_sdf) * 0.5, surface_proximity);
    
    // Apply with strength modulation
    let base_strength = brush.strength * 1.5;
    let influence_curve = influence * influence * (3.0 - 2.0 * influence);
    let smooth_strength = base_strength * influence_curve;
    let final_strength = saturate(smooth_strength + brush.smooth_bias * influence);
    
    return mix(current_sdf, morphed_sdf, final_strength);
}

// Note: get_neighbor_average function removed - functionality integrated into apply_smooth_brush

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = vec3<u32>(cache_metadata.dimensions);
    
    // Check bounds
    if (global_id.x >= dims.x || global_id.y >= dims.y || global_id.z >= dims.z) {
        return;
    }
    
    // Early Z-slice culling optimization
    // Calculate world Z position for this slice
    let cache_voxel_z = vec3<f32>(0.0, 0.0, f32(global_id.z) + 0.5);
    let world_z = cache_metadata.origin.z + cache_voxel_z.z * cache_metadata.voxel_size;
    
    // Early exit if this Z slice is too far from brush center
    let z_distance = abs(world_z - brush.position.z);
    if (z_distance > brush.radius * 1.1) { // 10% margin for safety
        return;
    }
    
    // ROBUST COORDINATE SYSTEM: Cache origin is corner, but sample at voxel centers  
    // cache[i] stores SDF sampled at: origin + (i + 0.5) * voxel_size
    let cache_voxel_center = vec3<f32>(global_id) + 0.5;
    let world_pos = cache_metadata.origin + cache_voxel_center * cache_metadata.voxel_size;
    
    // Calculate distance from brush center
    // Always use current brush position for influence calculation
    let distance = length(world_pos - brush.position);
    
    // Skip if outside brush radius
    if (distance > brush.radius) {
        return;
    }
    
    // Calculate brush influence
    let influence = calculate_falloff(distance, brush.radius, brush.falloff_type);
    
    // Skip if no influence (use small epsilon for early exit)
    if (influence <= 0.001) {
        return;
    }
    
    // Calculate linear index in cache
    let cache_idx = global_id.x + 
                    global_id.y * dims.x + 
                    global_id.z * dims.x * dims.y;
    
    // Get current SDF value
    let current_sdf = cache_data[cache_idx];
    
    // Apply brush operation
    var new_sdf = current_sdf;
    
    switch (brush.operation) {
        case BRUSH_SMOOTH: {
            new_sdf = apply_smooth_brush(current_sdf, influence, cache_idx, dims, world_pos, global_id);
        }
        default: {
            // Unknown brush type, no change
        }
    }
    
    // Hard clamp to prevent extreme values that could corrupt the SDF
    // Use a reasonable limit based on the maximum expected SDF values
    let max_sdf = cache_metadata.voxel_size * MAX_SDF_VALUE;
    new_sdf = clamp(new_sdf, -max_sdf, max_sdf);
    
    // Write back result
    cache_data[cache_idx] = new_sdf;
}
`,pi=`
${U.metadata}
${U.constants}
${U.utilities}

// Empty space SDF value (matches voxelHashMap.wgsl.js)
const VOXEL_EMPTY_SDF: f32 = 10.0;

// Maximum reasonable SDF value (in voxel units)
const MAX_SDF_VALUE: f32 = 50.0;

// Empty space threshold multiplier for consistency
const EMPTY_THRESHOLD_MULTIPLIER: f32 = 50.0;

// Brush operation types
const BRUSH_MOVE: u32 = 0u;
const BRUSH_BUMP: u32 = 1u;
const BRUSH_SMOOTH: u32 = 2u;

// Falloff types
const FALLOFF_CONSTANT: u32 = 0u;
const FALLOFF_LINEAR: u32 = 1u;
const FALLOFF_SMOOTH: u32 = 2u;
const FALLOFF_GAUSSIAN: u32 = 3u;
const FALLOFF_SHARP: u32 = 4u;

struct BrushParams {
    position: vec3<f32>,      // Brush center in world space
    radius: f32,              // Brush radius in world units
    strength: f32,            // Brush strength [0-1]
    operation: u32,           // Brush operation type
    falloff_type: u32,        // Falloff function type
    target_value: f32,        // For smooth operations
    normal: vec3<f32>,        // Surface normal (unused for MOVE)
    smooth_bias: f32,         // Smooth brush strength bias
    grab_original_pos: vec3<f32>, // For MOVE: where the grab started
    _padding2: u32,           // Align to 16 bytes
}

// Brush parameters (uniform)
@group(0) @binding(0) var<uniform> brush: BrushParams;

// Cache data (read-write)
@group(1) @binding(0) var<storage, read_write> cache_data: array<f32>;
@group(1) @binding(1) var<uniform> cache_metadata: CacheMetadata;

// Snapshot buffer for grab brushes (read-only)
@group(1) @binding(2) var<storage, read> snapshot_data: array<f32>;

// Calculate brush influence at a given distance
fn calculate_falloff(distance: f32, radius: f32, falloff_type: u32) -> f32 {
    let normalized_dist = saturate(distance / radius);
    
    switch (falloff_type) {
        case FALLOFF_CONSTANT: {
            return select(0.0, 1.0, distance <= radius);
        }
        case FALLOFF_LINEAR: {
            return 1.0 - normalized_dist;
        }
        case FALLOFF_SMOOTH: {
            // Smoother falloff that better matches the visual radius
            let t = 1.0 - normalized_dist;
            return t * t * (3.0 - 2.0 * t); // Standard smoothstep
        }
        case FALLOFF_GAUSSIAN: {
            // Gaussian falloff
            let sigma = 0.4; // Controls sharpness
            let t = normalized_dist / sigma;
            return exp(-0.5 * t * t);
        }
        case FALLOFF_SHARP: {
            // Sharp falloff near edges
            let t = 1.0 - normalized_dist;
            return t * t * t * t;
        }
        default: {
            return 0.0;
        }
    }
}

// Sample SDF from snapshot buffer at any world position using trilinear interpolation
fn sample_snapshot_sdf(world_pos: vec3<f32>) -> f32 {
    // Convert world position to cache local position
    let local_pos = world_pos - cache_metadata.origin;
    
    // Find which cache voxel contains this position
    let voxel_pos = (local_pos / cache_metadata.voxel_size) - 0.5;
    
    // Clamp to valid bounds
    let clamped_voxel_pos = clamp(voxel_pos, vec3<f32>(0.0), vec3<f32>(cache_metadata.dimensions) - vec3<f32>(1.001));
    
    // Trilinear interpolation using clamped position
    let voxel_coord = floor(clamped_voxel_pos);
    let fract_pos = fract(clamped_voxel_pos);
    
    // Get integer coordinates
    let x0 = u32(voxel_coord.x);
    let y0 = u32(voxel_coord.y);
    let z0 = u32(voxel_coord.z);
    
    let x1 = min(x0 + 1u, cache_metadata.dimensions.x - 1u);
    let y1 = min(y0 + 1u, cache_metadata.dimensions.y - 1u);
    let z1 = min(z0 + 1u, cache_metadata.dimensions.z - 1u);
    
    // Sample 8 corners from snapshot buffer
    let v000 = snapshot_data[x0 + y0 * cache_metadata.dimensions.x + z0 * cache_metadata.dimensions.x * cache_metadata.dimensions.y];
    let v100 = snapshot_data[x1 + y0 * cache_metadata.dimensions.x + z0 * cache_metadata.dimensions.x * cache_metadata.dimensions.y];
    let v010 = snapshot_data[x0 + y1 * cache_metadata.dimensions.x + z0 * cache_metadata.dimensions.x * cache_metadata.dimensions.y];
    let v110 = snapshot_data[x1 + y1 * cache_metadata.dimensions.x + z0 * cache_metadata.dimensions.x * cache_metadata.dimensions.y];
    let v001 = snapshot_data[x0 + y0 * cache_metadata.dimensions.x + z1 * cache_metadata.dimensions.x * cache_metadata.dimensions.y];
    let v101 = snapshot_data[x1 + y0 * cache_metadata.dimensions.x + z1 * cache_metadata.dimensions.x * cache_metadata.dimensions.y];
    let v011 = snapshot_data[x0 + y1 * cache_metadata.dimensions.x + z1 * cache_metadata.dimensions.x * cache_metadata.dimensions.y];
    let v111 = snapshot_data[x1 + y1 * cache_metadata.dimensions.x + z1 * cache_metadata.dimensions.x * cache_metadata.dimensions.y];
    
    // Trilinear interpolation
    let x00 = mix(v000, v100, fract_pos.x);
    let x10 = mix(v010, v110, fract_pos.x);
    let x01 = mix(v001, v101, fract_pos.x);
    let x11 = mix(v011, v111, fract_pos.x);
    
    let y_0 = mix(x00, x10, fract_pos.y);
    let y_1 = mix(x01, x11, fract_pos.y);
    
    return mix(y_0, y_1, fract_pos.z);
}

// Apply BUMP brush - snapshot-based displacement for better SDF preservation
fn apply_bump_brush(current_sdf: f32, world_pos: vec3<f32>, cache_idx: u32, influence: f32) -> f32 {
    // Get the original SDF value from snapshot
    let original_sdf = snapshot_data[cache_idx];
    
    // Calculate displacement along brush normal
    let displacement_distance = brush.strength * influence * cache_metadata.voxel_size;
    let displacement_vector = brush.normal * displacement_distance;
    
    // Sample from displaced position in snapshot for better SDF continuity
    let sample_pos = world_pos - displacement_vector;
    let displaced_sdf = sample_snapshot_sdf(sample_pos);
    
    // Preserve accumulative changes from original state
    let accumulated_change = current_sdf - original_sdf;
    let new_value = displaced_sdf + accumulated_change;
    
    // Apply influence-based blending
    let blend_factor = saturate(abs(brush.strength) * influence);
    
    // Add directional bias for push/pull effect
    let bias = sign(brush.strength) * influence * cache_metadata.voxel_size * 0.5;
    
    return mix(current_sdf, new_value - bias, blend_factor);
}

// Apply MOVE brush - cumulative deformation from original snapshot
fn apply_move_brush(current_sdf: f32, world_pos: vec3<f32>, cache_idx: u32) -> f32 {
    // Get the ORIGINAL SDF value at this position from snapshot
    let original_sdf = snapshot_data[cache_idx];
    
    // Calculate total movement from original grab position
    let movement = brush.position - brush.grab_original_pos;
    
    // For MOVE brush, we need to check two cases:
    // 1. Forward mapping: Is this voxel being moved FROM its current position?
    // 2. Reverse mapping: Is this voxel receiving data FROM another position?
    
    // Case 1: Forward mapping (this voxel is near the original grab position)
    let distance_from_original = length(world_pos - brush.grab_original_pos);
    let forward_influence = calculate_falloff(distance_from_original, brush.radius, brush.falloff_type);
    
    // Case 2: Reverse mapping (check if data is being moved TO this position)
    // We need to find where this voxel would have come from
    // If we're at position P and movement is M, the source would be at P - M
    // But we need to check if P - M was within the influence of the original grab
    var result_sdf = original_sdf;
    
    // Check multiple points along the reverse path to handle partial influences
    for (var i = 0; i <= 10; i++) {
        let t = f32(i) / 10.0;
        let check_movement = movement * t;
        let potential_source = world_pos - check_movement;
        let source_distance = length(potential_source - brush.grab_original_pos);
        let source_influence = calculate_falloff(source_distance, brush.radius, brush.falloff_type);
        
        if (source_influence > 0.01) {
            // This position would have been influenced at the source
            // Sample what was there originally
            let source_sdf = sample_snapshot_sdf(potential_source);
            
            // Blend based on influence and strength
            result_sdf = mix(result_sdf, source_sdf, brush.strength * source_influence);
        }
    }
    
    return result_sdf;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = vec3<u32>(cache_metadata.dimensions);
    
    // Check bounds
    if (global_id.x >= dims.x || global_id.y >= dims.y || global_id.z >= dims.z) {
        return;
    }
    
    // ROBUST COORDINATE SYSTEM: Cache origin is corner, but sample at voxel centers  
    // cache[i] stores SDF sampled at: origin + (i + 0.5) * voxel_size
    let cache_voxel_center = vec3<f32>(global_id) + 0.5;
    let world_pos = cache_metadata.origin + cache_voxel_center * cache_metadata.voxel_size;
    
    // For MOVE brush, we need a different approach:
    // We need to process ALL voxels in the cache because:
    // 1. Source voxels near the original position get moved
    // 2. Destination voxels near the current position receive data
    // 3. Voxels along the path might be affected by the smear
    
    // Calculate movement vector
    let movement = brush.position - brush.grab_original_pos;
    let movement_magnitude = length(movement);
    
    // Check distance from both original and current brush positions
    let distance_from_original = length(world_pos - brush.grab_original_pos);
    let distance_from_current = length(world_pos - brush.position);
    
    // This voxel could be affected if it's near either position
    // or anywhere along the movement path
    var min_distance = min(distance_from_original, distance_from_current);
    
    // For very large movements, we might need to check if the voxel
    // is near the line between original and current positions
    if (movement_magnitude > brush.radius * 2.0) {
        // Calculate distance to the line segment between positions
        let t = saturate(dot(world_pos - brush.grab_original_pos, movement) / (movement_magnitude * movement_magnitude));
        let closest_point = brush.grab_original_pos + t * movement;
        let distance_to_path = length(world_pos - closest_point);
        min_distance = min(min_distance, distance_to_path);
    }
    
    // For move brush, we process ALL voxels in the cache
    // The apply_move_brush function will determine what data should be at each position
    
    // We still calculate influence from original position for voxels that are being moved
    let influence = calculate_falloff(distance_from_original, brush.radius, brush.falloff_type);
    
    // Calculate linear index in cache
    let cache_idx = global_id.x + 
                    global_id.y * dims.x + 
                    global_id.z * dims.x * dims.y;
    
    // Get current SDF value
    let current_sdf = cache_data[cache_idx];
    
    // Apply brush operation
    var new_sdf = current_sdf;
    
    switch (brush.operation) {
        case BRUSH_MOVE: {
            new_sdf = apply_move_brush(current_sdf, world_pos, cache_idx);
        }
        case BRUSH_BUMP: {
            // Calculate influence for bump brush
            let distance_from_center = length(world_pos - brush.position);
            let bump_influence = calculate_falloff(distance_from_center, brush.radius, brush.falloff_type);
            
            if (bump_influence > 0.001) {
                new_sdf = apply_bump_brush(current_sdf, world_pos, cache_idx, bump_influence);
            }
        }
        default: {
            // Unsupported operation in grab shader
        }
    }
    
    // Hard clamp to prevent extreme values that could corrupt the SDF
    // Use a reasonable limit based on the maximum expected SDF values
    let max_sdf = cache_metadata.voxel_size * MAX_SDF_VALUE;
    new_sdf = clamp(new_sdf, -max_sdf, max_sdf);
    
    // Write back result
    cache_data[cache_idx] = new_sdf;
}
`,Se=globalThis.GPUBufferUsage||(typeof Se<"u"?Se:{}),$=globalThis.GPUShaderStage||(typeof $<"u"?$:{});class fi{constructor(e){this.gpu=e,this.pipeline=null,this.snapshotPipeline=null,this.brushParamsBuffer=null,this.brushBindGroup=null}async initialize(){const e=this.gpu.device.createShaderModule({label:"BrushPipeline.accumulatingShaderModule",code:hi}),t=this.gpu.device.createShaderModule({label:"BrushPipeline.snapshotShaderModule",code:pi});this.brushParamsBuffer=this.gpu.device.createBuffer({label:"BrushPipeline.brushParamsBuffer",size:64,usage:Se.UNIFORM|Se.COPY_DST});const i=this.gpu.device.createBindGroupLayout({label:"BrushPipeline.brushBindGroupLayout",entries:[{binding:0,visibility:$.COMPUTE,buffer:{type:"uniform"}}]}),s=this.gpu.device.createBindGroupLayout({label:"BrushPipeline.cacheBindGroupLayout",entries:[{binding:0,visibility:$.COMPUTE|$.FRAGMENT,buffer:{type:"storage"}},{binding:1,visibility:$.COMPUTE|$.FRAGMENT,buffer:{type:"uniform"}}]}),a=this.gpu.device.createBindGroupLayout({label:"BrushPipeline.grabCacheBindGroupLayout",entries:[{binding:0,visibility:$.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:$.COMPUTE,buffer:{type:"uniform"}},{binding:2,visibility:$.COMPUTE,buffer:{type:"read-only-storage"}}]}),n=this.gpu.device.createPipelineLayout({label:"BrushPipeline.pipelineLayout",bindGroupLayouts:[i,s]}),o=this.gpu.device.createPipelineLayout({label:"BrushPipeline.grabPipelineLayout",bindGroupLayouts:[i,a]});return this.pipeline=this.gpu.device.createComputePipeline({label:"BrushPipeline",layout:n,compute:{module:e,entryPoint:"main"}}),this.snapshotPipeline=this.gpu.device.createComputePipeline({label:"BrushPipeline.snapshot",layout:o,compute:{module:t,entryPoint:"main"}}),this.brushBindGroup=this.gpu.device.createBindGroup({label:"BrushPipeline.brushBindGroup",layout:i,entries:[{binding:0,resource:{buffer:this.brushParamsBuffer}}]}),this.pipeline}updateBrushParams(e){const t=new ArrayBuffer(64),i=new Float32Array(t),s=new Uint32Array(t);i[0]=e.position[0],i[1]=e.position[1],i[2]=e.position[2],i[3]=e.radius,i[4]=e.strength,s[5]=e.operation,s[6]=e.falloffType,i[7]=e.targetValue,i[8]=e.normal[0],i[9]=e.normal[1],i[10]=e.normal[2],s[11]=0,i[12]=e.grabOriginalPos?e.grabOriginalPos[0]:e.position[0],i[13]=e.grabOriginalPos?e.grabOriginalPos[1]:e.position[1],i[14]=e.grabOriginalPos?e.grabOriginalPos[2]:e.position[2],s[15]=0,this.gpu.device.queue.writeBuffer(this.brushParamsBuffer,0,t)}applyBrush(e,t){if(!this.pipeline)throw new Error("Pipeline not initialized. Call initialize() first.");this.updateBrushParams(t);const i=this.gpu.device.createCommandEncoder({label:"BrushPipeline.applyBrush"}),s=i.beginComputePass({label:"BrushPipeline.applyBrush.computePass"}),a=t.operation===ae.MOVE||t.operation===ae.BUMP,n=a?this.snapshotPipeline:this.pipeline;s.setPipeline(n),s.setBindGroup(0,this.brushBindGroup);const o=a?e.grabBindGroup:e.dataBindGroup;s.setBindGroup(1,o);const l=t.position,c=t.radius,p=e.origin,u=e.voxelSize,g=[Math.max(0,Math.floor((l[0]-c-p[0])/u)),Math.max(0,Math.floor((l[1]-c-p[1])/u)),Math.max(0,Math.floor((l[2]-c-p[2])/u))],v=[Math.min(e.dimensions.x,Math.ceil((l[0]+c-p[0])/u)),Math.min(e.dimensions.y,Math.ceil((l[1]+c-p[1])/u)),Math.min(e.dimensions.z,Math.ceil((l[2]+c-p[2])/u))];Math.max(1,v[0]-g[0]),Math.max(1,v[1]-g[1]),Math.max(1,v[2]-g[2]),e.updateDirtyBounds(g,v);const w=Math.ceil(e.dimensions.x/8),y=Math.ceil(e.dimensions.y/8),_=Math.ceil(e.dimensions.z/1);return s.dispatchWorkgroups(w,y,_),s.end(),this.gpu.device.queue.submit([i.finish()]),e.markDirty(),!0}destroy(){this.brushParamsBuffer&&(this.brushParamsBuffer.destroy(),this.brushParamsBuffer=null)}}const ae={MOVE:0,BUMP:1,ERODE:2},mi={},gi=`
${U.metadata}
${U.constants}
${De}
${Pe}

struct SyncStats {
    voxels_written: atomic<u32>,
    voxels_skipped: atomic<u32>,
    voxels_created: atomic<u32>,
    padding: u32,
};

@group(0) @binding(0) var<storage, read_write> voxel_hash_table: array<vec4<i32>>;
@group(0) @binding(1) var<storage, read_write> voxel_sdf_data: array<f32>;
@group(0) @binding(2) var<uniform> voxel_params: VoxelHashMapParams;

@group(1) @binding(0) var<storage, read_write> cache_data: array<f32>;
@group(1) @binding(1) var<uniform> cache_metadata: CacheMetadata;

@group(2) @binding(0) var<storage, read_write> sync_stats: SyncStats;

@group(3) @binding(0) var<storage, read_write> voxel_allocator: atomic<u32>;

// Find existing voxel slot - read-only operation
fn find_existing_voxel_slot(voxel_coord: vec3<i32>) -> i32 {
    let hash_index = hash_coords(voxel_coord.x, voxel_coord.y, voxel_coord.z);
    
    // Only look for existing voxels - never create new ones
    for (var i = 0u; i < MAX_HASH_PROBES; i++) {
        let slot = (hash_index + i) % HASH_TABLE_SIZE;
        let entry = voxel_hash_table[slot];
    
        // Found existing voxel - return its data index
        if entry.x == voxel_coord.x && entry.y == voxel_coord.y && entry.z == voxel_coord.z && entry.w != EMPTY_HASH_SLOT {
            return entry.w;
        }
    }
    
    // No existing voxel found
    return EMPTY_HASH_SLOT;
}

// Safe voxel creation with reduced race conditions for boundary extension
fn find_or_create_voxel_slot_safe(voxel_coord: vec3<i32>) -> i32 {
    let hash_index = hash_coords(voxel_coord.x, voxel_coord.y, voxel_coord.z);
    
    // First pass: check if voxel already exists (might have been created by another thread)
    for (var i = 0u; i < MAX_HASH_PROBES; i++) {
        let slot = (hash_index + i) % HASH_TABLE_SIZE;
        let entry = voxel_hash_table[slot];
    
        // Found existing voxel - return its data index
        if entry.x == voxel_coord.x && entry.y == voxel_coord.y && entry.z == voxel_coord.z && entry.w != EMPTY_HASH_SLOT {
            return entry.w;
        }
    }
    
    // Second pass: try to create with minimal race window
    var best_slot: i32 = -1;
    for (var i = 0u; i < MAX_HASH_PROBES; i++) {
        let slot = (hash_index + i) % HASH_TABLE_SIZE;
        let entry = voxel_hash_table[slot];
    
        // Found empty slot
        if entry.w == EMPTY_HASH_SLOT || entry.x == EMPTY_COORD_MARKER {
            best_slot = i32(slot);
            break;
        }
    }
    
    if best_slot != -1 {
        // Allocate data index first (atomic operation)
        let new_data_index = atomicAdd(&voxel_allocator, 1u);
        
        if new_data_index < voxel_params.max_voxels {
            // Double-check slot is still available
            let current_entry = voxel_hash_table[best_slot];
            if current_entry.w == EMPTY_HASH_SLOT || current_entry.x == EMPTY_COORD_MARKER {
                // Write entry carefully - coordinates first, then data index
                voxel_hash_table[best_slot].x = voxel_coord.x;
                voxel_hash_table[best_slot].y = voxel_coord.y;
                voxel_hash_table[best_slot].z = voxel_coord.z;
                voxel_hash_table[best_slot].w = i32(new_data_index);
                
                // Track creation
                atomicAdd(&sync_stats.voxels_created, 1u);
                return i32(new_data_index);
            } else {
                // Slot taken by another thread - release allocation
                atomicSub(&voxel_allocator, 1u);
            }
        } else {
            // No space in data array
            atomicSub(&voxel_allocator, 1u);
        }
    }

    return EMPTY_HASH_SLOT;
}

// Convert voxel coordinate back to world position (CENTER-BASED)
fn voxel_coord_to_world(voxel_coord: vec3<i32>, voxel_size: f32) -> vec3<f32> {
    return vec3<f32>(voxel_coord) * voxel_size + vec3<f32>(voxel_size * 0.5);
}

@compute @workgroup_size(8, 8, 4)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32
) {
    let cache_dims = cache_metadata.dimensions;
  
    // Simple, robust bounds check
    if global_id.x >= cache_dims.x || global_id.y >= cache_dims.y || global_id.z >= cache_dims.z {
        return;
    }
  
    // Calculate cache index and read SDF value
    let cache_index = global_id.x + global_id.y * cache_dims.x + global_id.z * cache_dims.x * cache_dims.y;
    let sdf_value = cache_data[cache_index];
  
    // ROBUST COORDINATE SYSTEM: Cache origin is corner, but sample at voxel centers
    // cache[i] stores SDF sampled at: origin + (i + 0.5) * voxel_size
    let cache_voxel_center = vec3<f32>(global_id) + 0.5;
    let world_pos = cache_metadata.origin + cache_voxel_center * cache_metadata.voxel_size;
    
    // CRITICAL FIX: Check if this cache voxel contains empty space data
    // Empty space has exactly VOXEL_EMPTY_SDF value (10.0)
    const VOXEL_EMPTY_SDF: f32 = 10.0;
    if sdf_value >= VOXEL_EMPTY_SDF - 0.001 {
        // This cache voxel contains empty space, skip sync
        atomicAdd(&sync_stats.voxels_skipped, 1u);
        return;
    }
    
    // CRITICAL: Match the narrow band width used during initial voxel construction
    // Initial construction uses 13 voxel narrow band - we must match this
    // to avoid stale voxels causing artifacts
    const NARROW_BAND_VOXELS = 5.0; // Reduced to prevent excessive voxel creation
    let sync_threshold = max(cache_metadata.voxel_size, voxel_params.voxel_size) * NARROW_BAND_VOXELS;
    
    // Only sync voxels that are near the surface (narrow band storage)
    if abs(sdf_value) > sync_threshold {
        atomicAdd(&sync_stats.voxels_skipped, 1u);
        return;
    }
  
    // FIXED RESOLUTION: Direct one-to-one mapping
    // No multi-resolution handling needed
    let voxel_coord = world_to_voxel_coord(world_pos, voxel_params.voxel_size);
    
    // Find or create voxel
    var data_index = find_existing_voxel_slot(voxel_coord);
  
    // If voxel doesn't exist, create it for surface areas that extend beyond initial boundaries
    if data_index == EMPTY_HASH_SLOT {
        // CRITICAL: Check if we're approaching voxel limit to prevent GPU overload
        let current_voxel_count = atomicLoad(&voxel_allocator);
        if current_voxel_count > voxel_params.max_voxels * 3u / 4u {
            // Too many voxels - skip creation to prevent GPU issues
            atomicAdd(&sync_stats.voxels_skipped, 1u);
            return;
        }
        
        // Use same narrow band threshold for consistency
        if abs(sdf_value) <= sync_threshold {
            // Use safe voxel creation for boundary extensions
            data_index = find_or_create_voxel_slot_safe(voxel_coord);

            if data_index == EMPTY_HASH_SLOT || data_index < 0 {
                // Creation failed - skip this voxel
                atomicAdd(&sync_stats.voxels_skipped, 1u);
                return;
            }
        } else {
            // Too far from surface - skip
            atomicAdd(&sync_stats.voxels_skipped, 1u);
            return;
        }
    }
  
    // OPTIMIZATION: Only write if value has changed
    // This prevents unnecessary memory writes and atomic operations
    let existing_value = voxel_sdf_data[data_index];
    if abs(existing_value - sdf_value) > 0.0001 {
        voxel_sdf_data[data_index] = sdf_value;
        atomicAdd(&sync_stats.voxels_written, 1u);
    } else {
        atomicAdd(&sync_stats.voxels_skipped, 1u);
    }
}
`,re=globalThis.GPUBufferUsage,Q=globalThis.GPUShaderStage;class _i{constructor(e){this.gpu=e,this.pipeline=null,this.statsBuffer=null,this.statsBindGroup=null,this.readbackBuffer=null,this.cachedAllocatorBindGroup=null}async initialize(){const e=this.gpu.device,t=e.createShaderModule({label:"SyncPipeline.shaderModule",code:gi});this.statsBuffer=e.createBuffer({label:"SyncPipeline.statsBuffer",size:16,usage:re.STORAGE|re.COPY_SRC|re.COPY_DST}),this.readbackBuffer=e.createBuffer({label:"SyncPipeline.readbackBuffer",size:16,usage:re.COPY_DST|re.MAP_READ});const i=e.createBindGroupLayout({label:"SyncPipeline.statsBindGroupLayout",entries:[{binding:0,visibility:Q.COMPUTE,buffer:{type:"storage"}}]});this.statsBindGroup=e.createBindGroup({label:"SyncPipeline.statsBindGroup",layout:i,entries:[{binding:0,resource:{buffer:this.statsBuffer}}]});const s=e.createBindGroupLayout({label:"SyncPipeline.cacheReadBindGroupLayout",entries:[{binding:0,visibility:Q.COMPUTE|Q.FRAGMENT,buffer:{type:"storage"}},{binding:1,visibility:Q.COMPUTE|Q.FRAGMENT,buffer:{type:"uniform"}}]}),a=e.createBindGroupLayout({label:"SyncPipeline.allocatorBindGroupLayout",entries:[{binding:0,visibility:Q.COMPUTE,buffer:{type:"storage"}}]}),n=e.createPipelineLayout({label:"SyncPipeline.layout",bindGroupLayouts:[this.gpu.bindGroupLayouts.voxelWrite,s,i,a]});return this.pipeline=e.createComputePipeline({label:"SyncPipeline",layout:n,compute:{module:t,entryPoint:"main"}}),this.pipeline}getCachedAllocatorBindGroup(e,t){if(this.cachedAllocatorBindGroup)return this.cachedAllocatorBindGroup;const i=[{binding:0,resource:{buffer:t.voxelAllocatorBuffer}}];return this.cachedAllocatorBindGroup=e.createBindGroup({label:"SyncPipeline.allocatorBindGroup",layout:this.pipeline.getBindGroupLayout(3),entries:i}),this.cachedAllocatorBindGroup}async syncToVoxels(e,t){var w;if(!this.pipeline)throw new Error("Pipeline not initialized");if(!e.isDirty)return{voxelsWritten:0,voxelsSkipped:0};const i=this.gpu.device;i.queue.writeBuffer(this.statsBuffer,0,new Uint32Array([0,0,0,0]));const s=i.createCommandEncoder({label:"SyncPipeline.syncToVoxels"}),a=s.beginComputePass({label:"SyncPipeline.computePass"});a.setPipeline(this.pipeline);let n;if(t.writeBindGroup)n=t.writeBindGroup;else if(t.createWriteBindGroup&&((w=this.gpu.bindGroupLayouts)!=null&&w.voxelWrite))n=t.createWriteBindGroup(this.gpu.bindGroupLayouts.voxelWrite);else throw new Error("Could not get voxel system write bind group");a.setBindGroup(0,n),a.setBindGroup(1,e.dataBindGroup),a.setBindGroup(2,this.statsBindGroup);const o=this.getCachedAllocatorBindGroup(i,t);a.setBindGroup(3,o);const l=Math.ceil(e.dimensions.x/8),c=Math.ceil(e.dimensions.y/8),p=Math.ceil(e.dimensions.z/4),u=65535;if(!(l>u||c>u||p>u))a.dispatchWorkgroups(l,c,p);else{const y=Math.min(l,u),_=Math.min(c,u),f=Math.min(p,u);a.dispatchWorkgroups(y,_,f)}a.end(),s.copyBufferToBuffer(this.statsBuffer,0,this.readbackBuffer,0,16),this.readbackBuffer.unmap(),i.queue.submit([s.finish()]);let v={voxelsWritten:0,voxelsSkipped:0,voxelsCreated:0};return e.clearDirty(),v}destroy(){this.statsBuffer&&(this.statsBuffer.destroy(),this.statsBuffer=null),this.readbackBuffer&&(this.readbackBuffer.destroy(),this.readbackBuffer=null),this.cachedAllocatorBindGroup=null,this.statsBindGroup=null,this.pipeline=null}}const vi=`
${U.metadata}
${U.constants}

// Empty voxel handling constants
const EMPTY_SDF_VALUE: f32 = 10.0;  // Match voxelHashMap.wgsl.js
const EMPTY_THRESHOLD_MULTIPLIER: f32 = 50.0;  // Consistent threshold
const MAX_REASONABLE_MULTIPLIER: f32 = 30.0;   // Max clamped distance

// JFA seed data - stores closest surface point info
struct JFASeed {
    closest_pos: vec3<f32>,     // Position of closest surface point
    distance: f32,              // Distance to that point (always positive)
    sign_at_surface: f32,       // Sign of SDF at the closest surface point
    is_valid: f32,              // 1.0 if valid seed, 0.0 otherwise
    _padding: f32,              // Padding for alignment
}

// Cache data
@group(0) @binding(0) var<storage, read_write> cache_data: array<f32>;
@group(0) @binding(1) var<uniform> cache_metadata: CacheMetadata;
@group(0) @binding(2) var<storage, read_write> jfa_seeds: array<JFASeed>; // JFA working buffer

// JFA parameters
struct JFAParams {
    step_size: u32,            // Current step size (power of 2)
    surface_threshold: f32,    // Threshold for detecting surface
    _padding1: u32,           
    _padding2: u32,
}

@group(1) @binding(0) var<uniform> params: JFAParams;

// Convert 3D position to linear index
fn pos_to_idx(pos: vec3<u32>, dims: vec3<u32>) -> u32 {
    return pos.x + pos.y * dims.x + pos.z * dims.x * dims.y;
}

// Convert linear index to 3D position
fn idx_to_pos(idx: u32, dims: vec3<u32>) -> vec3<u32> {
    let x = idx % dims.x;
    let y = (idx / dims.x) % dims.y;
    let z = idx / (dims.x * dims.y);
    return vec3<u32>(x, y, z);
}

// Get world position of voxel center
fn get_world_pos(voxel_pos: vec3<u32>) -> vec3<f32> {
    return cache_metadata.origin + (vec3<f32>(voxel_pos) + 0.5) * cache_metadata.voxel_size;
}

// Initialize pass - identify surface voxels and set as seeds
@compute @workgroup_size(4, 4, 4)
fn init(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = cache_metadata.dimensions;
    
    if (global_id.x >= dims.x || global_id.y >= dims.y || global_id.z >= dims.z) {
        return;
    }
    
    let pos = global_id;
    let idx = pos_to_idx(pos, dims);
    let world_pos = get_world_pos(pos);
    let sdf_value = cache_data[idx];
    
    var seed: JFASeed;
    
    // Initialize based on current SDF value
    let abs_sdf = abs(sdf_value);
    let sign_sdf = sign(sdf_value);
    
    // Check for empty space using consistent threshold
    let empty_threshold = cache_metadata.voxel_size * EMPTY_THRESHOLD_MULTIPLIER;
    // Empty voxels are marked with EMPTY_SDF_VALUE (10.0) in voxel system
    let is_empty = abs_sdf >= EMPTY_SDF_VALUE - 0.001 || abs_sdf > empty_threshold;
    
    if (is_empty) {
        // Empty space - no valid seed
        seed.closest_pos = vec3<f32>(1e10, 1e10, 1e10);
        seed.distance = 1e10;
        seed.sign_at_surface = 0.0;
        seed.is_valid = 0.0;
    } else if (abs_sdf < cache_metadata.voxel_size * 1.2) {
        // Close to surface - high confidence seed
        seed.closest_pos = world_pos;
        seed.distance = abs_sdf;
        seed.sign_at_surface = sign_sdf;
        seed.is_valid = 1.0;
    } else if (abs_sdf < cache_metadata.voxel_size * 3.0) {
        // Near surface - still create good seeds
        seed.closest_pos = world_pos;
        seed.distance = abs_sdf;
        seed.sign_at_surface = sign_sdf;
        seed.is_valid = 0.8;
    } else {
        // Interior/exterior but not empty
        seed.closest_pos = world_pos;
        seed.distance = abs_sdf;
        seed.sign_at_surface = sign_sdf;
        seed.is_valid = 0.3;
    }
    
    jfa_seeds[idx] = seed;
}

// JFA step - propagate closest surface information
@compute @workgroup_size(4, 4, 4)
fn jfa_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = cache_metadata.dimensions;
    
    if (global_id.x >= dims.x || global_id.y >= dims.y || global_id.z >= dims.z) {
        return;
    }
    
    let pos = global_id;
    let idx = pos_to_idx(pos, dims);
    let world_pos = get_world_pos(pos);
    
    var best_seed = jfa_seeds[idx];
    let step = i32(params.step_size);
    
    // For empty space, we need to be more aggressive in finding seeds
    let is_empty_space = best_seed.is_valid < 0.01 && best_seed.distance > cache_metadata.voxel_size * EMPTY_THRESHOLD_MULTIPLIER;
    
    // Check 3x3x3 neighborhood at current step size
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                // Skip center (0,0,0) as it's already our current best
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }
                
                // Calculate neighbor position
                let neighbor_pos = vec3<i32>(pos) + vec3<i32>(dx, dy, dz) * step;
                
                // Check bounds
                if (neighbor_pos.x < 0 || neighbor_pos.x >= i32(dims.x) ||
                    neighbor_pos.y < 0 || neighbor_pos.y >= i32(dims.y) ||
                    neighbor_pos.z < 0 || neighbor_pos.z >= i32(dims.z)) {
                    continue;
                }
                
                let neighbor_idx = pos_to_idx(vec3<u32>(neighbor_pos), dims);
                let neighbor_seed = jfa_seeds[neighbor_idx];
                
                // Only propagate from valid seeds
                if (neighbor_seed.is_valid < 0.01) {
                    continue;
                }
                
                // Calculate distance from current voxel to neighbor's closest surface point
                let dist_to_seed = length(world_pos - neighbor_seed.closest_pos);
                
                // For empty space, accept any valid seed. Otherwise require improvement.
                var should_update = false;
                
                if (is_empty_space && neighbor_seed.is_valid > 0.01) {
                    // Empty space - accept any valid seed
                    should_update = true;
                } else {
                    // Normal case - only update if significantly closer
                    let improvement_threshold = cache_metadata.voxel_size * 0.1;
                    should_update = dist_to_seed < best_seed.distance - improvement_threshold;
                }
                
                if (should_update) {
                    best_seed.closest_pos = neighbor_seed.closest_pos;
                    best_seed.distance = dist_to_seed;
                    best_seed.sign_at_surface = neighbor_seed.sign_at_surface;
                    best_seed.is_valid = max(best_seed.is_valid, neighbor_seed.is_valid);
                }
            }
        }
    }
    
    jfa_seeds[idx] = best_seed;
}

// Final pass - compute proper signed distances
@compute @workgroup_size(4, 4, 4)
fn finalize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = cache_metadata.dimensions;
    
    if (global_id.x >= dims.x || global_id.y >= dims.y || global_id.z >= dims.z) {
        return;
    }
    
    let pos = global_id;
    let idx = pos_to_idx(pos, dims);
    let seed = jfa_seeds[idx];
    let original_sdf = cache_data[idx];
    let original_sign = sign(original_sdf);
    let original_abs = abs(original_sdf);
    
    // Conservative correction strategy with consistent thresholds
    let max_reasonable_distance = cache_metadata.voxel_size * MAX_REASONABLE_MULTIPLIER;
    let empty_threshold = cache_metadata.voxel_size * EMPTY_THRESHOLD_MULTIPLIER;
    
    // Detect if original was empty space (using same logic as init)
    let was_empty = original_abs >= EMPTY_SDF_VALUE - 0.001 || original_abs > empty_threshold;
    
    // Check if we have a valid JFA result
    if (seed.is_valid > 0.01 && seed.distance < 1e9) {
        let jfa_distance = seed.distance * original_sign;
        
        if (was_empty) {
            // Empty space that found a seed - use JFA result but ensure proper empty marking
            if (seed.distance > empty_threshold) {
                // Still far from any surface - maintain empty marker
                cache_data[idx] = sign(original_sdf) * EMPTY_SDF_VALUE;
            } else {
                // Close enough to surface - use actual distance
                let clamped_distance = min(seed.distance, max_reasonable_distance);
                // Determine sign from the seed's surface  
                let final_sign = select(-1.0, 1.0, seed.sign_at_surface > 0.0);
                cache_data[idx] = final_sign * clamped_distance;
            }
        } else {
            // Not empty - apply conservative correction
            let difference = abs(jfa_distance - original_sdf);
            var needs_correction = false;
            var correction_strength = 0.0;
            
            // Original value is problematic
            if (original_abs > max_reasonable_distance) {
                needs_correction = true;
                correction_strength = 1.0;
            }
            // Significant inconsistency detected
            else if (difference > cache_metadata.voxel_size * 5.0 && 
                     original_abs > cache_metadata.voxel_size * 5.0) {
                needs_correction = true;
                correction_strength = saturate(difference / (cache_metadata.voxel_size * 20.0));
            }
            
            if (needs_correction) {
                // Apply correction with damping
                let damping = 0.7;
                cache_data[idx] = mix(original_sdf, jfa_distance, correction_strength * damping);
            } else {
                // Keep original
                cache_data[idx] = original_sdf;
            }
        }
    } else {
        // No valid JFA result
        if (was_empty) {
            // Empty space with no seed - maintain empty marker
            cache_data[idx] = sign(original_sdf) * EMPTY_SDF_VALUE;
        } else if (original_abs > max_reasonable_distance) {
            // Not empty but too large - clamp to reasonable value
            cache_data[idx] = original_sign * max_reasonable_distance;
        } else {
            // Keep original if reasonable
            cache_data[idx] = original_sdf;
        }
    }
}

// Gradient smoothing pass to fix view-dependent artifacts
@compute @workgroup_size(4, 4, 4)
fn smooth_gradients(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = cache_metadata.dimensions;
    
    if (global_id.x >= dims.x || global_id.y >= dims.y || global_id.z >= dims.z) {
        return;
    }
    
    let pos = global_id;
    let idx = pos_to_idx(pos, dims);
    let current_sdf = cache_data[idx];
    let current_abs = abs(current_sdf);
    
    // Skip empty voxels - they should maintain their empty status
    if (current_abs >= EMPTY_SDF_VALUE - 0.001) {
        return;
    }
    
    // Only smooth gradients very close to surface to avoid artifacts
    if (current_abs > cache_metadata.voxel_size * 2.0) {
        return;
    }
    
    // Sample 6-connected neighbors
    var gradient_sum = vec3<f32>(0.0);
    var valid_samples = 0.0;
    
    // X neighbors
    if (pos.x > 0u) {
        let left_idx = idx - 1u;
        let left_sdf = cache_data[left_idx];
        // Skip empty neighbors
        if (abs(left_sdf) < EMPTY_SDF_VALUE - 0.001) {
            gradient_sum.x += (current_sdf - left_sdf) / cache_metadata.voxel_size;
            valid_samples += 1.0;
        }
    }
    if (pos.x < dims.x - 1u) {
        let right_idx = idx + 1u;
        let right_sdf = cache_data[right_idx];
        // Skip empty neighbors
        if (abs(right_sdf) < EMPTY_SDF_VALUE - 0.001) {
            gradient_sum.x += (right_sdf - current_sdf) / cache_metadata.voxel_size;
            valid_samples += 1.0;
        }
    }
    
    // Y neighbors
    if (pos.y > 0u) {
        let down_idx = idx - dims.x;
        let down_sdf = cache_data[down_idx];
        // Skip empty neighbors
        if (abs(down_sdf) < EMPTY_SDF_VALUE - 0.001) {
            gradient_sum.y += (current_sdf - down_sdf) / cache_metadata.voxel_size;
            valid_samples += 1.0;
        }
    }
    if (pos.y < dims.y - 1u) {
        let up_idx = idx + dims.x;
        let up_sdf = cache_data[up_idx];
        // Skip empty neighbors
        if (abs(up_sdf) < EMPTY_SDF_VALUE - 0.001) {
            gradient_sum.y += (up_sdf - current_sdf) / cache_metadata.voxel_size;
            valid_samples += 1.0;
        }
    }
    
    // Z neighbors
    if (pos.z > 0u) {
        let back_idx = idx - dims.x * dims.y;
        let back_sdf = cache_data[back_idx];
        // Skip empty neighbors
        if (abs(back_sdf) < EMPTY_SDF_VALUE - 0.001) {
            gradient_sum.z += (current_sdf - back_sdf) / cache_metadata.voxel_size;
            valid_samples += 1.0;
        }
    }
    if (pos.z < dims.z - 1u) {
        let front_idx = idx + dims.x * dims.y;
        let front_sdf = cache_data[front_idx];
        // Skip empty neighbors
        if (abs(front_sdf) < EMPTY_SDF_VALUE - 0.001) {
            gradient_sum.z += (front_sdf - current_sdf) / cache_metadata.voxel_size;
            valid_samples += 1.0;
        }
    }
    
    if (valid_samples > 0.0) {
        // Average gradient
        let avg_gradient = gradient_sum / (valid_samples / 3.0);
        let gradient_magnitude = length(avg_gradient);
        
        // Check if gradient is reasonable (should be close to 1 for SDF)
        if (gradient_magnitude > 0.5 && gradient_magnitude < 2.0) {
            // Gradient seems valid - no correction needed
            return;
        }
    }
    
    // If we get here, gradient might be problematic
    // Apply gentle Laplacian smoothing
    var neighbor_sum = 0.0;
    var neighbor_count = 0.0;
    
    // 6-connected neighbors (skip empty voxels)
    if (pos.x > 0u) {
        let n_sdf = cache_data[idx - 1u];
        if (abs(n_sdf) < EMPTY_SDF_VALUE - 0.001) {
            neighbor_sum += n_sdf;
            neighbor_count += 1.0;
        }
    }
    if (pos.x < dims.x - 1u) {
        let n_sdf = cache_data[idx + 1u];
        if (abs(n_sdf) < EMPTY_SDF_VALUE - 0.001) {
            neighbor_sum += n_sdf;
            neighbor_count += 1.0;
        }
    }
    if (pos.y > 0u) {
        let n_sdf = cache_data[idx - dims.x];
        if (abs(n_sdf) < EMPTY_SDF_VALUE - 0.001) {
            neighbor_sum += n_sdf;
            neighbor_count += 1.0;
        }
    }
    if (pos.y < dims.y - 1u) {
        let n_sdf = cache_data[idx + dims.x];
        if (abs(n_sdf) < EMPTY_SDF_VALUE - 0.001) {
            neighbor_sum += n_sdf;
            neighbor_count += 1.0;
        }
    }
    if (pos.z > 0u) {
        let n_sdf = cache_data[idx - dims.x * dims.y];
        if (abs(n_sdf) < EMPTY_SDF_VALUE - 0.001) {
            neighbor_sum += n_sdf;
            neighbor_count += 1.0;
        }
    }
    if (pos.z < dims.z - 1u) {
        let n_sdf = cache_data[idx + dims.x * dims.y];
        if (abs(n_sdf) < EMPTY_SDF_VALUE - 0.001) {
            neighbor_sum += n_sdf;
            neighbor_count += 1.0;
        }
    }
    
    if (neighbor_count > 0.0) {
        let neighbor_avg = neighbor_sum / neighbor_count;
        // Extremely gentle smoothing to preserve details
        cache_data[idx] = mix(current_sdf, neighbor_avg, 0.05);
    }
}

// Alternative robust method: Direct surface detection and distance computation
// This can be used for extreme cases where JFA might miss thin features
@compute @workgroup_size(4, 4, 4)
fn robust_redistance(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = cache_metadata.dimensions;
    
    if (global_id.x >= dims.x || global_id.y >= dims.y || global_id.z >= dims.z) {
        return;
    }
    
    let pos = global_id;
    let idx = pos_to_idx(pos, dims);
    let world_pos = get_world_pos(pos);
    let current_sdf = cache_data[idx];
    let current_abs = abs(current_sdf);
    
    // Skip empty voxels that are correctly marked
    if (current_abs >= EMPTY_SDF_VALUE - 0.001 && current_abs <= EMPTY_SDF_VALUE + 0.001) {
        return; // Already properly marked as empty
    }
    
    // Define search radius based on how extreme the current value is
    let search_radius = min(u32(current_abs / cache_metadata.voxel_size + 3.0), 20u);
    
    var closest_surface_dist = 1e10;
    var found_surface = false;
    var surface_sign = sign(current_sdf);
    
    // Search in expanding spherical shells for efficiency
    for (var r = 1u; r <= search_radius; r++) {
        var found_in_shell = false;
        
        // Check all voxels at distance r
        for (var dz = -i32(r); dz <= i32(r); dz++) {
            for (var dy = -i32(r); dy <= i32(r); dy++) {
                for (var dx = -i32(r); dx <= i32(r); dx++) {
                    // Only check voxels on the current shell
                    let max_coord = max(max(abs(dx), abs(dy)), abs(dz));
                    if (max_coord != i32(r)) {
                        continue;
                    }
                    
                    let check_pos = vec3<i32>(pos) + vec3<i32>(dx, dy, dz);
                    
                    // Bounds check
                    if (check_pos.x < 0 || check_pos.x >= i32(dims.x) ||
                        check_pos.y < 0 || check_pos.y >= i32(dims.y) ||
                        check_pos.z < 0 || check_pos.z >= i32(dims.z)) {
                        continue;
                    }
                    
                    let check_idx = pos_to_idx(vec3<u32>(check_pos), dims);
                    let check_sdf = cache_data[check_idx];
                    
                    // Check for zero crossing between current and checked voxel
                    if (sign(current_sdf) != sign(check_sdf)) {
                        let check_world_pos = get_world_pos(vec3<u32>(check_pos));
                        let dist = length(world_pos - check_world_pos);
                        
                        // Interpolate to find more accurate surface position
                        let t = abs(current_sdf) / (abs(current_sdf) + abs(check_sdf));
                        let surface_dist = dist * t;
                        
                        if (surface_dist < closest_surface_dist) {
                            closest_surface_dist = surface_dist;
                            found_surface = true;
                            found_in_shell = true;
                        }
                    }
                }
            }
        }
        
        // Early exit if we found surface in this shell
        if (found_in_shell) {
            break;
        }
    }
    
    // Update SDF value
    if (found_surface) {
        cache_data[idx] = surface_sign * closest_surface_dist;
    } else {
        // No surface found within search radius
        let search_distance = f32(search_radius) * cache_metadata.voxel_size;
        if (search_distance >= cache_metadata.voxel_size * EMPTY_THRESHOLD_MULTIPLIER) {
            // Far enough to be considered empty space
            cache_data[idx] = surface_sign * EMPTY_SDF_VALUE;
        } else {
            // Not quite empty but no surface found - clamp to search radius
            cache_data[idx] = surface_sign * search_distance;
        }
    }
}
`,ze=globalThis.GPUBufferUsage,fe=globalThis.GPUShaderStage;class yi{constructor(e){this.gpu=e,this.initPipeline=null,this.jfaStepPipeline=null,this.finalizePipeline=null,this.smoothGradientsPipeline=null,this.robustPipeline=null,this.paramsBuffer=null,this.jfaSeedsBuffer=null,this.paramsBindGroup=null}async initialize(){const e=this.gpu.device.createShaderModule({label:"RedistancingJFAPipeline.shaderModule",code:vi});this.paramsBuffer=this.gpu.device.createBuffer({label:"RedistancingJFAPipeline.paramsBuffer",size:16,usage:ze.UNIFORM|ze.COPY_DST});const t=this.gpu.device.createBindGroupLayout({label:"RedistancingJFAPipeline.cacheBindGroupLayout",entries:[{binding:0,visibility:fe.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:fe.COMPUTE,buffer:{type:"uniform"}},{binding:2,visibility:fe.COMPUTE,buffer:{type:"storage"}}]}),i=this.gpu.device.createBindGroupLayout({label:"RedistancingJFAPipeline.paramsBindGroupLayout",entries:[{binding:0,visibility:fe.COMPUTE,buffer:{type:"uniform"}}]}),s=this.gpu.device.createPipelineLayout({label:"RedistancingJFAPipeline.layout",bindGroupLayouts:[t,i]});this.initPipeline=this.gpu.device.createComputePipeline({label:"RedistancingJFAPipeline.init",layout:s,compute:{module:e,entryPoint:"init"}}),this.jfaStepPipeline=this.gpu.device.createComputePipeline({label:"RedistancingJFAPipeline.jfaStep",layout:s,compute:{module:e,entryPoint:"jfa_step"}}),this.finalizePipeline=this.gpu.device.createComputePipeline({label:"RedistancingJFAPipeline.finalize",layout:s,compute:{module:e,entryPoint:"finalize"}}),this.smoothGradientsPipeline=this.gpu.device.createComputePipeline({label:"RedistancingJFAPipeline.smoothGradients",layout:s,compute:{module:e,entryPoint:"smooth_gradients"}}),this.robustPipeline=this.gpu.device.createComputePipeline({label:"RedistancingJFAPipeline.robust",layout:s,compute:{module:e,entryPoint:"robust_redistance"}}),this.paramsBindGroup=this.gpu.device.createBindGroup({label:"RedistancingJFAPipeline.paramsBindGroup",layout:i,entries:[{binding:0,resource:{buffer:this.paramsBuffer}}]})}createJFASeedsBuffer(e){this.jfaSeedsBuffer&&this.jfaSeedsBuffer.destroy();const t=32;this.jfaSeedsBuffer=this.gpu.device.createBuffer({label:"RedistancingJFAPipeline.jfaSeedsBuffer",size:e*t,usage:ze.STORAGE})}updateParams(e,t){const i=new ArrayBuffer(16),s=new DataView(i);s.setUint32(0,e,!0),s.setFloat32(4,t,!0),this.gpu.device.queue.writeBuffer(this.paramsBuffer,0,i)}async applyJFARedistancing(e,t={}){const i=e.dimensions.x*e.dimensions.y*e.dimensions.z;(!this.jfaSeedsBuffer||this.jfaSeedsBufferSize!==i)&&(this.createJFASeedsBuffer(i),this.jfaSeedsBufferSize=i);const s=this.gpu.device.createBindGroup({label:"RedistancingJFAPipeline.cacheBindGroup",layout:this.initPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.dataBuffer}},{binding:1,resource:{buffer:e.metadataBuffer}},{binding:2,resource:{buffer:this.jfaSeedsBuffer}}]}),a=this.gpu.device.createCommandEncoder({label:"RedistancingJFAPipeline.apply"}),n=Math.ceil(e.dimensions.x/4),o=Math.ceil(e.dimensions.y/4),l=Math.ceil(e.dimensions.z/4),c=t.surfaceThreshold||e.voxelSize*1.2;this.updateParams(0,c);const p=a.beginComputePass({label:"JFA.init"});p.setPipeline(this.initPipeline),p.setBindGroup(0,s),p.setBindGroup(1,this.paramsBindGroup),p.dispatchWorkgroups(n,o,l),p.end();const u=Math.max(e.dimensions.x,e.dimensions.y,e.dimensions.z),g=Math.ceil(Math.log2(u));for(let y=0;y<g;y++){const _=Math.pow(2,g-y-1);this.updateParams(_,c);const f=a.beginComputePass({label:`JFA.step${y}`});f.setPipeline(this.jfaStepPipeline),f.setBindGroup(0,s),f.setBindGroup(1,this.paramsBindGroup),f.dispatchWorkgroups(n,o,l),f.end()}this.updateParams(1,c);const v=a.beginComputePass({label:"JFA.extraFineStep"});v.setPipeline(this.jfaStepPipeline),v.setBindGroup(0,s),v.setBindGroup(1,this.paramsBindGroup),v.dispatchWorkgroups(n,o,l),v.end();const w=a.beginComputePass({label:"JFA.finalize"});if(w.setPipeline(this.finalizePipeline),w.setBindGroup(0,s),w.setBindGroup(1,this.paramsBindGroup),w.dispatchWorkgroups(n,o,l),w.end(),t.skipGradientSmoothing!==!0){const y=a.beginComputePass({label:"JFA.smoothGradients"});y.setPipeline(this.smoothGradientsPipeline),y.setBindGroup(0,s),y.setBindGroup(1,this.paramsBindGroup),y.dispatchWorkgroups(n,o,l),y.end()}if(t.useRobustFallback){const y=a.beginComputePass({label:"JFA.robust"});y.setPipeline(this.robustPipeline),y.setBindGroup(0,s),y.setBindGroup(1,this.paramsBindGroup),y.dispatchWorkgroups(n,o,l),y.end()}return this.gpu.device.queue.submit([a.finish()]),e.markDirty(),!0}async applyGradientSmoothingOnly(e){if(!this.smoothGradientsPipeline)throw new Error("Pipeline not initialized. Call initialize() first.");const t=this.gpu.device.createBindGroup({label:"RedistancingJFAPipeline.cacheBindGroup",layout:this.smoothGradientsPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.dataBuffer}},{binding:1,resource:{buffer:e.metadataBuffer}},{binding:2,resource:{buffer:this.jfaSeedsBuffer}}]}),i=this.gpu.device.createCommandEncoder({label:"RedistancingJFAPipeline.smoothOnly"}),s=Math.ceil(e.dimensions.x/4),a=Math.ceil(e.dimensions.y/4),n=Math.ceil(e.dimensions.z/4),o=e.voxelSize*1.2;this.updateParams(0,o);const l=i.beginComputePass({label:"JFA.smoothGradientsOnly"});return l.setPipeline(this.smoothGradientsPipeline),l.setBindGroup(0,t),l.setBindGroup(1,this.paramsBindGroup),l.dispatchWorkgroups(s,a,n),l.end(),this.gpu.device.queue.submit([i.finish()]),e.markDirty(),!0}destroy(){this.paramsBuffer&&(this.paramsBuffer.destroy(),this.paramsBuffer=null),this.jfaSeedsBuffer&&(this.jfaSeedsBuffer.destroy(),this.jfaSeedsBuffer=null)}}const xi=`
${U.metadata}
${U.constants}

// Empty space handling constants (match redistancingJFA.wgsl.js)
const EMPTY_SDF_VALUE: f32 = 10.0;
const EMPTY_THRESHOLD_MULTIPLIER: f32 = 50.0;

// Cache data (read-write)
@group(0) @binding(0) var<storage, read_write> cache_data: array<f32>;
@group(0) @binding(1) var<uniform> cache_metadata: CacheMetadata;
@group(0) @binding(2) var<storage, read_write> cache_data_temp: array<f32>;  // Temporary buffer for ping-pong

// Redistancing parameters
struct RedistancingParams {
    max_distance: f32,       // Maximum distance to propagate
    epsilon: f32,            // Surface thickness for sign determination
    _padding1: u32,          // Padding for alignment
    _padding2: u32,          // Padding for alignment
}

@group(1) @binding(0) var<uniform> params: RedistancingParams;


// Get neighbor value with bounds checking
fn get_neighbor(pos: vec3<i32>, offset: vec3<i32>, dims: vec3<u32>, current_value: f32) -> f32 {
    let neighbor_pos = pos + offset;
    
    // Check bounds
    if (neighbor_pos.x < 0 || neighbor_pos.x >= i32(dims.x) ||
        neighbor_pos.y < 0 || neighbor_pos.y >= i32(dims.y) ||
        neighbor_pos.z < 0 || neighbor_pos.z >= i32(dims.z)) {
        return params.max_distance; // Return max distance for out of bounds
    }
    
    let idx = neighbor_pos.x + neighbor_pos.y * i32(dims.x) + neighbor_pos.z * i32(dims.x * dims.y);
    let neighbor_val = cache_data_temp[idx];
    
    // If neighbor is empty, return max distance to avoid propagating from empty space
    if (abs(neighbor_val) >= EMPTY_SDF_VALUE - 0.001) {
        return params.max_distance;
    }
    
    return neighbor_val; // Read from temp buffer during sweeps
}

// Solve the Eikonal equation using Godunov's scheme with proper upwinding
// Returns the new distance value for this voxel
fn solve_eikonal(pos: vec3<i32>, dims: vec3<u32>, sign_val: f32, current_value: f32) -> f32 {
    let h = cache_metadata.voxel_size;
    
    // Get neighbor values in each direction
    let nx_minus = get_neighbor(pos, vec3<i32>(-1, 0, 0), dims, current_value);
    let nx_plus = get_neighbor(pos, vec3<i32>(1, 0, 0), dims, current_value);
    let ny_minus = get_neighbor(pos, vec3<i32>(0, -1, 0), dims, current_value);
    let ny_plus = get_neighbor(pos, vec3<i32>(0, 1, 0), dims, current_value);
    let nz_minus = get_neighbor(pos, vec3<i32>(0, 0, -1), dims, current_value);
    let nz_plus = get_neighbor(pos, vec3<i32>(0, 0, 1), dims, current_value);
    
    // Use Godunov upwind scheme
    // For each axis, choose the neighbor with same sign and smaller absolute value
    var nx = params.max_distance;
    var ny = params.max_distance;
    var nz = params.max_distance;
    
    // X axis
    if (sign(nx_minus) == sign_val && abs(nx_minus) < nx) { nx = abs(nx_minus); }
    if (sign(nx_plus) == sign_val && abs(nx_plus) < nx) { nx = abs(nx_plus); }
    
    // Y axis  
    if (sign(ny_minus) == sign_val && abs(ny_minus) < ny) { ny = abs(ny_minus); }
    if (sign(ny_plus) == sign_val && abs(ny_plus) < ny) { ny = abs(ny_plus); }
    
    // Z axis
    if (sign(nz_minus) == sign_val && abs(nz_minus) < nz) { nz = abs(nz_minus); }
    if (sign(nz_plus) == sign_val && abs(nz_plus) < nz) { nz = abs(nz_plus); }
    
    // Sort the neighbor values (ascending order)
    var a = min(min(nx, ny), nz);
    var b = max(min(nx, ny), min(max(nx, ny), nz));
    var c = max(max(nx, ny), nz);
    
    // Solve for distance using sorted values
    var d = params.max_distance;
    
    // Case 1: Use only smallest neighbor
    if (a < params.max_distance) {
        d = a + h;
    }
    
    // Case 2: Try to use two smallest neighbors
    if (b < params.max_distance && d > b) {
        let discriminant = 2.0 * h * h - (a - b) * (a - b);
        if (discriminant >= 0.0) {
            let d2 = 0.5 * (a + b + sqrt(discriminant));
            if (d2 >= max(a, b)) {
                d = d2;
            }
        }
    }
    
    // Case 3: Try to use all three neighbors
    if (c < params.max_distance && d > c) {
        let p = a + b + c;
        let q = a * a + b * b + c * c;
        let discriminant = p * p - 3.0 * (q - h * h);
        if (discriminant >= 0.0) {
            let d3 = (p + sqrt(discriminant)) / 3.0;
            if (d3 >= max(max(a, b), c)) {
                d = d3;
            }
        }
    }
    
    // Apply sign and clamp to max distance
    return sign_val * min(d, params.max_distance);
}


fn process_voxel(pos: vec3<i32>, dims: vec3<u32>) {
    let idx = pos.x + pos.y * i32(dims.x) + pos.z * i32(dims.x * dims.y);
    
    // Get current value
    let current = cache_data_temp[idx];
    let abs_current = abs(current);
    
    // Skip empty voxels - they should maintain their empty status
    if (abs_current >= EMPTY_SDF_VALUE - 0.001) {
        cache_data[idx] = current;
        return;
    }
    
    // Ensure sign_val is never zero for stable upwind scheme
    let sign_val = select(select(-1.0, 1.0, current >= 0.0), sign(current), abs_current > params.epsilon);
    
    // Define narrowband for processing
    let narrowband = cache_metadata.voxel_size * 10.0;
    
    // Only process if not at interface and within narrowband
    if (abs_current > params.epsilon && abs_current < narrowband) {
        // Solve Eikonal equation
        let new_value = solve_eikonal(pos, dims, sign_val, current);
        
        // More conservative update with adaptive relaxation
        let distance_from_interface = abs(current) / cache_metadata.voxel_size;
        let adaptive_relaxation = select(0.9, 0.6, distance_from_interface > 3.0);
        let updated = current + adaptive_relaxation * (new_value - current);
        
        // Only update if improvement is significant and stable
        let improvement = abs(current) - abs(updated);
        if (improvement > cache_metadata.voxel_size * 0.01) {
            cache_data[idx] = updated;
        } else {
            cache_data[idx] = current;
        }
    } else {
        // At interface or outside narrowband - preserve value
        cache_data[idx] = current;
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = cache_metadata.dimensions;
    let total_voxels = dims.x * dims.y * dims.z;
    
    // Check bounds
    if (global_id.x >= total_voxels) {
        return;
    }
    
    // Convert 1D index to 3D position
    let pos = vec3<i32>(
        i32(global_id.x % dims.x),
        i32((global_id.x / dims.x) % dims.y),
        i32(global_id.x / (dims.x * dims.y))
    );
    
    process_voxel(pos, dims);
}

// Initialize pass - prepare data for fast sweeping
@compute @workgroup_size(64)
fn init(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_voxels = cache_metadata.dimensions.x * 
                       cache_metadata.dimensions.y * 
                       cache_metadata.dimensions.z;
    
    if (global_id.x >= total_voxels) {
        return;
    }
    
    let val = cache_data[global_id.x];
    let abs_val = abs(val);
    
    // Check if this is an empty voxel (consistent with JFA)
    let empty_threshold = cache_metadata.voxel_size * EMPTY_THRESHOLD_MULTIPLIER;
    let is_empty = abs_val >= EMPTY_SDF_VALUE - 0.001 || abs_val > empty_threshold;
    
    // Define narrowband width - make it adaptive to voxel size
    let narrowband = cache_metadata.voxel_size * 8.0;
    
    // Initialize based on distance to interface
    if (is_empty) {
        // Empty voxel - preserve empty marker
        cache_data_temp[global_id.x] = sign(val) * EMPTY_SDF_VALUE;
    } else if (abs_val < params.epsilon) {
        // At interface - keep exact value
        cache_data_temp[global_id.x] = val;
    } else if (abs_val < narrowband) {
        // Inside narrowband - use current value as initial guess
        cache_data_temp[global_id.x] = val;
    } else {
        // Outside narrowband - clamp to narrowband distance with proper sign
        // Use a slightly smaller clamp to avoid hard boundaries
        let clamped_distance = narrowband * 0.9;
        cache_data_temp[global_id.x] = sign(val) * clamped_distance;
    }
}

// Copy pass - copies result back to temp buffer for next iteration
@compute @workgroup_size(64)
fn copy_back(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_voxels = cache_metadata.dimensions.x * 
                       cache_metadata.dimensions.y * 
                       cache_metadata.dimensions.z;
    
    if (global_id.x >= total_voxels) {
        return;
    }
    
    cache_data_temp[global_id.x] = cache_data[global_id.x];
}
`,me=globalThis.GPUBufferUsage,ge=globalThis.GPUShaderStage;class bi{constructor(e){this.gpu=e,this.pipeline=null,this.initPipeline=null,this.copyPipeline=null,this.paramsBuffer=null,this.tempBuffer=null,this.paramsBindGroup=null}async initialize(){const e=this.gpu.device.createShaderModule({label:"RedistancingPipeline.shaderModule",code:xi});this.paramsBuffer=this.gpu.device.createBuffer({label:"RedistancingPipeline.paramsBuffer",size:16,usage:me.UNIFORM|me.COPY_DST});const t=this.gpu.device.createBindGroupLayout({label:"RedistancingPipeline.cacheBindGroupLayout",entries:[{binding:0,visibility:ge.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:ge.COMPUTE,buffer:{type:"uniform"}},{binding:2,visibility:ge.COMPUTE,buffer:{type:"storage"}}]}),i=this.gpu.device.createBindGroupLayout({label:"RedistancingPipeline.paramsBindGroupLayout",entries:[{binding:0,visibility:ge.COMPUTE,buffer:{type:"uniform"}}]}),s=this.gpu.device.createPipelineLayout({label:"RedistancingPipeline.pipelineLayout",bindGroupLayouts:[t,i]});return this.initPipeline=this.gpu.device.createComputePipeline({label:"RedistancingPipeline.init",layout:s,compute:{module:e,entryPoint:"init"}}),this.pipeline=this.gpu.device.createComputePipeline({label:"RedistancingPipeline.main",layout:s,compute:{module:e,entryPoint:"main"}}),this.copyPipeline=this.gpu.device.createComputePipeline({label:"RedistancingPipeline.copy",layout:s,compute:{module:e,entryPoint:"copy_back"}}),this.paramsBindGroup=this.gpu.device.createBindGroup({label:"RedistancingPipeline.paramsBindGroup",layout:i,entries:[{binding:0,resource:{buffer:this.paramsBuffer}}]}),this.pipeline}ensureTempBuffer(e){const t=e.dimensions.x*e.dimensions.y*e.dimensions.z*4;(!this.tempBuffer||this.tempBuffer.size<t)&&(this.tempBuffer&&this.tempBuffer.destroy(),this.tempBuffer=this.gpu.device.createBuffer({label:"RedistancingPipeline.tempBuffer",size:t,usage:me.STORAGE|me.COPY_DST}))}async applyRedistancing(e,t,i={}){if(!this.pipeline||!this.copyPipeline||!this.initPipeline)throw new Error("Pipeline not initialized. Call initialize() first.");this.ensureTempBuffer(e);const s=this.gpu.device.createBindGroup({label:"RedistancingPipeline.cacheBindGroup",layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.dataBuffer}},{binding:1,resource:{buffer:e.metadataBuffer}},{binding:2,resource:{buffer:this.tempBuffer}}]}),a=e.dimensions.x*e.dimensions.y*e.dimensions.z,n=65535,o=64,l=n*o,c=i.maxDistance,p=i.epsilon,u=new ArrayBuffer(16),g=new Uint32Array(u),v=new Float32Array(u);v[0]=c,v[1]=p,g[2]=0,g[3]=0,this.gpu.device.queue.writeBuffer(this.paramsBuffer,0,u);const w=this.gpu.device.createCommandEncoder({label:"RedistancingPipeline.initialize"}),y=w.beginComputePass();y.setPipeline(this.initPipeline),y.setBindGroup(0,s),y.setBindGroup(1,this.paramsBindGroup);const _=Math.ceil(a/o);if(_<=n)y.dispatchWorkgroups(_);else{let m=0;for(;m<a;){const S=Math.min(l,a-m),x=Math.ceil(S/o);y.dispatchWorkgroups(x),m+=S}}y.end(),this.gpu.device.queue.submit([w.finish()]);const f=[];for(let m=0;m<t;m++){const S=this.gpu.device.createCommandEncoder({label:`RedistancingPipeline.sweep${m}`}),x=S.beginComputePass({label:`RedistancingPipeline.sweep${m}`});x.setPipeline(this.pipeline),x.setBindGroup(0,s),x.setBindGroup(1,this.paramsBindGroup);const T=Math.ceil(a/o);if(T<=n)x.dispatchWorkgroups(T);else{let E=0;for(;E<a;){const C=Math.min(l,a-E),z=Math.ceil(C/o);x.dispatchWorkgroups(z),E+=C}}x.end();const A=S.beginComputePass({label:`RedistancingPipeline.copy${m}`});if(A.setPipeline(this.copyPipeline),A.setBindGroup(0,s),A.setBindGroup(1,this.paramsBindGroup),T<=n)A.dispatchWorkgroups(T);else{let E=0;for(;E<a;){const C=Math.min(l,a-E),z=Math.ceil(C/o);A.dispatchWorkgroups(z),E+=C}}A.end(),f.push(S.finish())}return this.gpu.device.queue.submit(f),await this.gpu.device.queue.onSubmittedWorkDone(),!0}destroy(){this.paramsBuffer&&(this.paramsBuffer.destroy(),this.paramsBuffer=null),this.tempBuffer&&(this.tempBuffer.destroy(),this.tempBuffer=null)}}const _e=globalThis.GPUBufferUsage,N=globalThis.GPUShaderStage;class wi{constructor(e,t){this.device=e,this.viewport=t,this.isActive=!1,this.needsSync=!1,this.gpu={device:this.device,bindGroupLayouts:{}},this.adaptiveCache=null,this.allocator=new li({maxCacheSize:d.sculpting.adaptiveCache.maxCacheSize,baseVoxelSize:d.voxel.voxelSize}),this.brushStrategies={MOVE:new ci(this.allocator),BUMP:new qe(this.allocator),ERODE:new qe(this.allocator)},this.fillPipeline=new di(this.gpu),this.brushPipeline=new fi(this.gpu),this.syncPipeline=new _i(this.gpu),this.redistancingJFAPipeline=new yi(this.gpu),this.redistancingPipeline=new bi(this.gpu),this.currentBrushType=null,this.currentBrushPosition=M(),this.cameraPosition=M(),this.cameraDistance=0,this.cacheBuffer=null,this.metadataBuffer=null,this.voxelReadOnlyBindGroupLayout=null,this.voxelBindGroupLayout=null,this.cacheBindGroupLayout=null,this.enableRedistancing=!0,this.lastActiveTime=Date.now(),this.cacheTimeout=2e3,this.memoryPressureCallback=null,this.evictionTimer=null,this.initializeCompatibilityBuffers()}initializeCompatibilityBuffers(){var e;(!this.cacheBuffer||this.cacheBuffer===((e=this.adaptiveCache)==null?void 0:e.dataBuffer))&&(this.cacheBuffer=this.device.createBuffer({label:"Adaptive Cache Compatibility Buffer",size:d.bufferSizes.MIN_DUMMY_CACHE_SIZE,usage:_e.STORAGE|_e.COPY_DST})),this.metadataBuffer||(this.metadataBuffer=this.device.createBuffer({label:"Adaptive Cache Metadata",size:d.bufferSizes.CACHE_METADATA_SIZE,usage:_e.UNIFORM|_e.COPY_DST})),this.updateMetadata()}updateMetadata(){const e=new ArrayBuffer(d.bufferSizes.CACHE_METADATA_SIZE),t=new Float32Array(e),i=new Uint32Array(e);t.fill(0),this.adaptiveCache&&this.isActive?(t[0]=this.adaptiveCache.origin[0],t[1]=this.adaptiveCache.origin[1],t[2]=this.adaptiveCache.origin[2],t[3]=this.adaptiveCache.voxelSize,i[4]=this.adaptiveCache.dimensions.x,i[5]=this.adaptiveCache.dimensions.y,i[6]=this.adaptiveCache.dimensions.z,i[7]=1):i[7]=0,this.device.queue.writeBuffer(this.metadataBuffer,0,e)}async createPipelines(e){this.createBindGroupLayouts(),await this.initializePipelines()}createBindGroupLayouts(){this.voxelReadOnlyBindGroupLayout=this.device.createBindGroupLayout({label:"VoxelHashMap Read-Only Layout",entries:[{binding:0,visibility:N.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:N.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:N.COMPUTE,buffer:{type:"uniform"}}]}),this.voxelBindGroupLayout=this.device.createBindGroupLayout({label:"VoxelHashMap Read-Write Layout",entries:[{binding:0,visibility:N.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:N.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:N.COMPUTE,buffer:{type:"uniform"}}]}),this.cacheBindGroupLayout=this.device.createBindGroupLayout({label:"Adaptive Cache Bind Group Layout",entries:[{binding:0,visibility:N.COMPUTE|N.FRAGMENT,buffer:{type:"storage"}},{binding:1,visibility:N.COMPUTE|N.FRAGMENT,buffer:{type:"uniform"}}]}),this.gpu.bindGroupLayouts={voxelRead:this.voxelReadOnlyBindGroupLayout,voxelWrite:this.voxelBindGroupLayout,cacheRead:this.adaptiveCache?this.adaptiveCache.dataBindGroupLayout:this.cacheBindGroupLayout}}async initializePipelines(){await this.fillPipeline.initialize(),await this.brushPipeline.initialize(),await this.syncPipeline.initialize(),await this.redistancingJFAPipeline.initialize(),await this.redistancingPipeline.initialize()}async applyRedistancing(e,t={}){const i={...t,skipGradientSmoothing:!1,surfaceThreshold:e.voxelSize*1.2,useRobustFallback:!1};await this.redistancingJFAPipeline.applyJFARedistancing(e,i);const s={maxDistance:e.voxelSize*3,epsilon:e.voxelSize*.5};return await this.redistancingPipeline.applyRedistancing(e,1,s),await this.redistancingJFAPipeline.applyGradientSmoothingOnly(e),!0}activate(e){const t=e.view;xe(this.cameraPosition,-t[12],-t[13],-t[14]),this.isActive=!0,this.lastActiveTime=Date.now(),this.startEvictionTimer(),this.updateMetadata()}startEvictionTimer(){this.clearEvictionTimer(),this.evictionTimer=setTimeout(()=>{this.isActive&&!this.needsSync&&(console.log("[AdaptiveCacheIntegration] Cache timeout - evicting inactive cache"),this.deactivate())},this.cacheTimeout)}clearEvictionTimer(){this.evictionTimer&&(clearTimeout(this.evictionTimer),this.evictionTimer=null)}async deactivate(){var e;if(this.isActive=!1,this.needsSync=!1,this.clearEvictionTimer(),this.adaptiveCache){this.initializeCompatibilityBuffers(),this.updateMetadata(),(e=this.viewport)!=null&&e.bindGroupSystem&&this.viewport.bindGroupSystem.invalidate("viewport_adaptive_cache"),this.onCacheAllocated&&this.onCacheAllocated();const t=this.adaptiveCache;this.adaptiveCache=null;const i=this.gpu.device.createCommandEncoder();this.gpu.device.queue.submit([i.finish()]),await this.gpu.device.queue.onSubmittedWorkDone(),t.destroy()}}isSculptingDistanceValid(e,t){const i=e.position,s=G(t[0],t[1],t[2]),a=Je(i,s),n=Ge.adaptiveCache.maxSculptingDistance;return!(a>n)}shouldReallocateCache(e,t){if(!this.adaptiveCache||!e||Math.abs(this.adaptiveCache.voxelSize-e.voxelSize)>.001)return!0;const i=this.adaptiveCache.voxelSize*2,s=this.adaptiveCache.origin,a=M();if(ne(a,s,[this.adaptiveCache.dimensions.x,this.adaptiveCache.dimensions.y,this.adaptiveCache.dimensions.z],this.adaptiveCache.voxelSize),t[0]<s[0]+i||t[0]>a[0]-i||t[1]<s[1]+i||t[1]>a[1]-i||t[2]<s[2]+i||t[2]>a[2]-i)return!0;const n=this.adaptiveCache.dimensions.x*this.adaptiveCache.dimensions.y*this.adaptiveCache.dimensions.z,o=e.dimensions.x*e.dimensions.y*e.dimensions.z;return Math.abs(o-n)/n>.5}async checkMemoryPressure(){if(!navigator.gpu||!navigator.gpu.requestAdapter)return!1;try{const e=await navigator.gpu.requestAdapter();if(e&&e.limits){const i=(this.adaptiveCache?this.adaptiveCache.dimensions.x*this.adaptiveCache.dimensions.y*this.adaptiveCache.dimensions.z:0)*4/(1024*1024);if(i>100)return console.log(`[AdaptiveCacheIntegration] High memory usage: ${i.toFixed(1)}MB`),!0}}catch{}return!1}async allocateCacheForBrush(e,t,i,s,a){const n=G(t[0],t[1],t[2]);if(!this.isSculptingDistanceValid(s,n))return!1;if(this.adaptiveCache&&this.isActive){const c=this.allocator.allocateScreenAligned(n,s,a,i);if(!c)throw new Error("[AdaptiveCacheIntegration] Failed to calculate new allocation for comparison");if(!this.shouldReallocateCache(c,n))return!0;await this.deactivate()}const o=this.allocator.allocateScreenAligned(n,s,a,i);if(!o)throw new Error("[AdaptiveCacheIntegration] Failed to get allocation from strategy");const l=o.worldBounds?Z(o.worldBounds.min):M();return this.adaptiveCache=new oi(this.gpu,o.dimensions,o.voxelSize,l),this.gpu.bindGroupLayouts.cacheRead=this.adaptiveCache.dataBindGroupLayout,this.cacheBuffer=this.adaptiveCache.dataBuffer,this.currentBrushType=e,ce(this.currentBrushPosition,n),this.isActive=!0,this.updateMetadata(),this.onCacheAllocated&&this.onCacheAllocated(),!0}async updateBrushPosition(e){this.adaptiveCache&&ce(this.currentBrushPosition,e)}async fillFromVoxels(e){return this.adaptiveCache?await this.adaptiveCache.fillFromVoxels(e,this.fillPipeline.getPipeline()):!1}async applyInitialRedistancing(){if(!this.adaptiveCache)return!1;if(this.enableRedistancing&&this.adaptiveCache.voxelSize<=this.allocator.baseVoxelSize*3){const e={surfaceThreshold:this.adaptiveCache.voxelSize*1.2,useRobustFallback:!1};await this.applyRedistancing(this.adaptiveCache,e)}return!0}async applyBrush(e){if(!this.adaptiveCache)return!1;this.lastActiveTime=Date.now(),this.startEvictionTimer();const t=e.type.toUpperCase(),i={MOVE:ae.MOVE,BUMP:ae.BUMP,ERODE:ae.ERODE},s={position:e.position,radius:e.radius,strength:e.strength,operation:i[t],falloffType:e.falloffType||mi.SMOOTH,targetValue:0,normal:e.normal,grabOriginalPos:e.grabOriginalPos},a=await this.adaptiveCache.applyBrush(s,this.brushPipeline);if(a){this.needsSync=!0;const n=s.operation===ae.MOVE;if(this.enableRedistancing&&!n&&this.adaptiveCache.voxelSize<=this.allocator.baseVoxelSize*3){const o={surfaceThreshold:this.adaptiveCache.voxelSize*1.2,useRobustFallback:!1};await this.applyRedistancing(this.adaptiveCache,o)}}return a}async captureSnapshot(){if(!this.adaptiveCache)return!1;const e=this.gpu.device.createCommandEncoder({label:"AdaptiveCacheIntegration.captureSnapshot"});return this.adaptiveCache.captureSnapshot(e),this.gpu.device.queue.submit([e.finish()]),await this.gpu.device.queue.onSubmittedWorkDone(),!0}async syncToVoxels(e){if(!this.adaptiveCache||!this.needsSync)return{voxelsWritten:0,voxelsSkipped:0};if(!this.adaptiveCache.isDirty)return this.needsSync=!1,{voxelsWritten:0,voxelsSkipped:0};if(this.viewport&&this.viewport.camera){const i=this.viewport.camera.position,s=this.adaptiveCache.origin,a=s[0]-i[0],n=s[1]-i[1],o=s[2]-i[2];if(Math.sqrt(a*a+n*n+o*o)>20)return this.needsSync=!1,{voxelsWritten:0,voxelsSkipped:0}}const t=await this.adaptiveCache.syncToVoxels(e,this.syncPipeline);return this.needsSync=!1,t}onBrushRelease(){if(!this.currentBrushType)return;const e=this.currentBrushType.toUpperCase(),t=this.brushStrategies[e];if(!t)throw new Error(`No strategy found for brush type: ${this.currentBrushType} (${e})`);t.onMouseUp&&t.onMouseUp(),t.needsFinalCommit&&t.needsFinalCommit()&&this.adaptiveCache&&this.adaptiveCache.isDirty&&(this.needsSync=!0)}needsResize(){return!1}resize(){}destroy(){var e;this.adaptiveCache&&(this.adaptiveCache.destroy(),this.adaptiveCache=null),this.cacheBuffer&&this.cacheBuffer!==((e=this.adaptiveCache)==null?void 0:e.dataBuffer)&&this.cacheBuffer.destroy(),this.metadataBuffer&&this.metadataBuffer.destroy(),this.fillPipeline.destroy(),this.brushPipeline.destroy(),this.syncPipeline.destroy(),this.redistancingJFAPipeline.destroy(),this.redistancingPipeline.destroy()}}const K={MOVE:0,BUMP:1,ERODE:2};class Si{constructor(e,t){this.name="SculptingSystem",this.device=e,this.viewport=t,this.initialized=!1,this.workgroupOptimizer=new Xt(t.workgroupLimits),this.workgroupSizes={linear:this.workgroupOptimizer.getLinearWorkgroupSize(),voxel3D:this.workgroupOptimizer.get3DWorkgroupSize()},this.voxelHashMap=new Wt(e),this.state=new ei,this.brushManager=new It,this.brushManager.setSculptingState(this.state),this.brushManager.setOperation(K.MOVE),this.state.updateBrushSettings("type",{type:K.MOVE}),this.initializeGPUResources(),this.inputHandler=new si(this.viewport,this),this.editingCache=new wi(e,t),this.executor=new ti(e,this.editingCache,this.brushManager),this.editingCache.onCacheAllocated=()=>{this.viewport&&this.viewport.setupAdaptiveCacheResources&&this.viewport.setupAdaptiveCacheResources()}}async initialize(){this.initialized||(this.state.needsBindGroupRefresh&&(this.bindGroupManager.refreshBindGroups(this.voxelHashMap,this.bufferManager),this.state.needsBindGroupRefresh=!1),await this.editingCache.createPipelines(this.brushManager),this.initialized=!0)}getBindGroup(e){return this.bindGroupManager.getBindGroup(e)}get showBrushPreview(){return this.state.showBrushPreview}set showBrushPreview(e){this.state.showBrushPreview=e}get brushPreviewPosition(){return this.state.brushPreviewPosition}set brushPreviewPosition(e){this.state.brushPreviewPosition=e}get lastRaycastHit(){return this.state.lastRaycastHit}set lastRaycastHit(e){this.state.lastRaycastHit=e}get lastRayOrigin(){return this.state.lastRayOrigin}set lastRayOrigin(e){this.state.lastRayOrigin=e}get lastRayDirection(){return this.state.lastRayDirection}set lastRayDirection(e){this.state.lastRayDirection=e}initializeGPUResources(){this.pipelineInitializer=new Kt(this.device,this.workgroupSizes);const{layouts:e,pipelines:t}=this.pipelineInitializer.initializePipelines();this.layouts=e,this.pipelines=t,this.bufferManager=new Jt(this.device),this.bufferManager.initializeBuffers(),this.bindGroupManager=new Qt(this.viewport.bindGroupSystem),this.bindGroupManager.setLayouts(this.layouts),this.bindGroupManager.initializeBindGroups(),this.raycastHandler=new ii(this.device,this.bufferManager,this,this.pipelines),this.raycastHandler.setViewport(this.viewport)}refreshSculptingBindGroups(){this.bindGroupManager.refreshBindGroups(this.voxelHashMap,this.bufferManager),this.state.needsBindGroupRefresh=!1}shouldHandleMouseEvent(e){return this.inputHandler.shouldHandleMouseEvent(e)}getCurrentBrush(){return this.brushManager.getOperationName()}getAvailableBrushes(){return[{type:K.MOVE,name:"Move",description:"Grab and move surface"},{type:K.BUMP,name:"Bump",description:"Push/pull surface"},{type:K.ERODE,name:"Erode",description:"Average neighbors"}]}async performSculpting(){this.state.needsBindGroupRefresh&&(this.refreshSculptingBindGroups(),this.state.needsBindGroupRefresh=!1),this.state.setPendingOperation(this.prepareSculptOperation()),this.state.isProcessingSculpt||this.processPendingSculpt()}async processPendingSculpt(){if(this.state.pendingSculptOperation){for(this.state.setProcessingState(!0);this.state.pendingSculptOperation;){const e=this.state.pendingSculptOperation;this.state.clearPendingOperation(),await this.executeSculpting(e)}this.state.setProcessingState(!1)}}async executeSculpting(e){const t={viewport:this.viewport,bufferManager:this.bufferManager,bindGroupManager:this.bindGroupManager,voxelHashMap:this.voxelHashMap};await this.executor.execute(e,t),this.state.incrementOperationCount(),this.state.setProcessingState(!1)}async syncEditingCache(){await this.executor.syncEditingCache(this.voxelHashMap,this.viewport)}refreshViewportBindGroups(){}setBrushType(e){let t=e;if(typeof e=="string"&&(t=parseInt(e,10),isNaN(t)))throw new Error(`Cannot parse brush type "${e}" as a number`);const i=Object.values(K);if(!i.includes(t))throw new Error(`Invalid brush type: ${t}. Valid types are: ${i.join(", ")}`);if(this.brushManager.setOperation(t),this.state.updateBrushSettings("type",{type:t}),this.showBrushPreview&&this.viewport){const s=this.state.getBrushSettings();this.viewport.updateBrushPreview(s.position,s.radius,s.normal,this.lastRaycastHit?this.lastRaycastHit.distance:1)}this.emitBrushChange({brushName:this.brushManager.getOperationName()})}setBrushRadius(e){this.state.updateBrushSettings("radius",{radius:e}),this.brushManager.setRadius(e),this.emitBrushChange()}setBrushStrength(e){this.state.updateBrushSettings("strength",{strength:e}),this.brushManager.setStrength(e),this.emitBrushChange()}setMouseThrottleFPS(e){this.inputHandler&&this.inputHandler.setMouseThrottleFPS(e)}emitBrushChange(e={}){const t=this.state.getBrushSettings();L.emit(k.BRUSH_CHANGE,{type:t.type,radius:t.radius,strength:t.strength,...e})}getBrushOperations(){return Object.entries(K).map(([e,t])=>({type:t,name:e.charAt(0)+e.slice(1).toLowerCase(),description:this.brushManager.getOperationName()===e?"Active":""}))}getCurrentBrushOperation(){return this.brushManager.state.operation}prepareSculptOperation(){const e=this.inputHandler.getState(),t={pressure:e.tabletPressure,modifiers:e.modifierKeys},i=this.brushManager.getGPUParams(),s=this.state.getBrushSettings();if(i.position=s.position,i.normal=s.normal,t.modifiers&&t.modifiers.ctrl&&(i.strength=-i.strength),this.viewport&&this.viewport.camera&&i.position){const l=this.viewport.camera.position,c=i.position,p=Math.sqrt(Math.pow(c[0]-l[0],2)+Math.pow(c[1]-l[1],2)+Math.pow(c[2]-l[2],2)),u=Math.min(p/2,3);i.erodeBias=(i.erodeBias||.15)*u}i.position?!Array.isArray(i.position)&&!ArrayBuffer.isView(i.position)?i.position=new Float32Array([0,0,0]):i.position.length<3&&(i.position=new Float32Array([0,0,0])):i.position=new Float32Array([0,0,0]),i.normal?!Array.isArray(i.normal)&&!ArrayBuffer.isView(i.normal)?i.normal=new Float32Array([0,1,0]):i.normal.length<3&&(i.normal=new Float32Array([0,1,0])):i.normal=new Float32Array([0,1,0]);let a,n;a=Array.from(i.position),n=Array.from(i.normal);let o;return i.grabOriginalPos?o=Array.from(i.grabOriginalPos):o=a,{position:a,radius:i.radius,strength:i.strength,operation:i.operation,falloffType:i.falloffType,targetValue:i.targetValue,normal:n,grabOriginalPos:o}}destroy(){this.voxelHashMap&&(this.voxelHashMap.destroy&&this.voxelHashMap.destroy(),this.voxelHashMap=null),this.bufferManager&&(this.bufferManager.destroy(),this.bufferManager=null),this.initialized=!1}}function Pi(r){r.updateBrushPreview=function(e,t,i,s){if(!e)throw new Error("updateBrushPreview called with null or undefined position");Array.isArray(e)?this.brushPreviewPosition=e:this.brushPreviewPosition=[e[0],e[1],e[2]],this.brushPreviewRadius=t,this.brushPreviewDistance=s,i?Array.isArray(i)?this.brushPreviewNormal=i:this.brushPreviewNormal=[i[0],i[1],i[2]]:this.brushPreviewNormal=[0,1,0],this.showBrushPreview=!0,this.isDirty=!0},r.hideBrushPreview=function(){this.showBrushPreview=!1,this.isDirty=!0}}async function Bi(r){const{webgpu:e,device:t,canvas:i}=r;r.resourceManager=$t(t),r.gBufferManager=new Yt(t,i.width,i.height),r.pipelineManager=new qt(e),r.uniformManager=new Bt(t),r.bindGroupSystem=new Zt(t)}async function Mi(r){const{device:e}=r;r.sculptingSystem=new Si(e,r),Pi(r),await r.sculptingSystem.initialize(),await r.sculptingSystem.voxelHashMap.waitForInit(),r.sculptingSystem.voxelHashMap.addTestGeometry()}async function Ei(r){r.defineBindGroups(),await r.createPipeline(),r.setupBindGroupResources(),r.createBrushPreviewPipeline()}function Ai(r){L.on(k.SCULPT_END,()=>{r.isDirty=!0})}async function Ci(r,e){const i=await(await fetch(e)).blob(),s=await createImageBitmap(i),a=r.createTexture({size:[s.width,s.height,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});return r.queue.copyExternalImageToTexture({source:s},{texture:a},[s.width,s.height]),a}class Ri{constructor(e){this.canvas=document.getElementById(e),this.webgpu=new Ze,this.pipeline=null,this.brushPreviewPipeline=null,this.gBufferManager=null,this.matcapTexture=null,this.depthTexture=null,this.pipelineManager=null,this.resourceManager=null,this.camera=new xt,this.input=new bt(this.canvas),this.navigation={x:0,y:0,zoom:0,panX:0,panY:0},this.settings={lensDistortionEnabled:!1},this.isDirty=!0,this.animationFrameId=null,this.resizeTimeout=null,this.showBrushPreview=!1,this.brushPreviewPosition=[0,0,0],this.brushPreviewRadius=.5,this.brushPreviewNormal=[0,1,0],this.brushPreviewUniformData=new ArrayBuffer(256),this.brushPreviewFloatView=new Float32Array(this.brushPreviewUniformData)}async init(){if(!navigator.gpu){this.showError("WebGPU is not supported in this browser/environment");return}await this.webgpu.initialize(this.canvas,[],{maxStorageBuffersPerShaderStage:10}),this.device=this.webgpu.device,this.presentationFormat=this.webgpu.presentationFormat,this.workgroupLimits=this.webgpu.workgroupLimits,await Bi(this),await Mi(this),this.matcapTexture=await Ci(this.device,"./data/matcap.png"),await Ei(this),Ai(this),this.camera.update(this.canvas.width,this.canvas.height),this.setupUniforms(),this.updateUniforms(),this.gui=new Ht(this),this.setupInputHandlers(),this.setupWindowEvents(),this.renderLoop()}defineBindGroups(){this.bindGroupSystem.defineBindGroup("viewport_uniforms",se.UNIFORMS,[{binding:0,type:b.BUFFER},{binding:1,type:b.TEXTURE},{binding:2,type:b.SAMPLER}]),this.bindGroupSystem.defineBindGroup("viewport_voxel_hashmap",se.SPATIAL_DATA,[{binding:0,type:b.BUFFER},{binding:1,type:b.BUFFER},{binding:2,type:b.BUFFER}])}setupBindGroupResources(){var e;if(this.camera.update(this.canvas.width,this.canvas.height),this.setupUniforms(),this.updateUniforms(),this.uniformManager.updateBuffers(this.device.queue),this.bindGroupSystem.setResources("viewport_uniforms",{0:{resource:this.viewportBuffer.gpuBuffer,type:b.BUFFER},1:{resource:this.matcapTexture,type:b.TEXTURE},2:{resource:this.resourceManager.getLinearSampler(),type:b.SAMPLER}}),(e=this.sculptingSystem)!=null&&e.voxelHashMap){const t=this.sculptingSystem.voxelHashMap;this.bindGroupSystem.setResources("viewport_voxel_hashmap",{0:{resource:t.hashTableBuffer,type:b.BUFFER},1:{resource:t.voxelDataBuffer,type:b.BUFFER},2:{resource:t.paramsBuffer,type:b.BUFFER}})}this.bindGroupSystem.defineBindGroup("viewport_adaptive_cache",2,[{binding:0,type:b.BUFFER},{binding:1,type:b.BUFFER}]),this.setupAdaptiveCacheResources()}setupAdaptiveCacheResources(){var e;if((e=this.sculptingSystem)!=null&&e.editingCache){const t=this.sculptingSystem.editingCache;t.cacheBuffer&&t.metadataBuffer&&this.bindGroupSystem.setResources("viewport_adaptive_cache",{0:{resource:t.cacheBuffer,type:b.BUFFER},1:{resource:t.metadataBuffer,type:b.BUFFER}})}}async createPipeline(){await this.createIntegratedPipeline()}async createIntegratedPipeline(){const e=[Ce.uniform,Ce.voxelHashMap,Ce.adaptiveCache],t={name:"IntegratedRaymarcher",vertexShader:He,fragmentShader:He,vertexEntryPoint:"vs_main",bindGroupLayouts:e,fragmentState:{targets:[{format:this.presentationFormat}]}},i=await this.pipelineManager.createPipeline(t);this.pipeline=i.pipeline,this.bindGroupSystem.registerLayout("viewport_uniforms_layout",i.bindGroupLayouts[0]),this.bindGroupSystem.registerLayout("viewport_voxel_hashmap_layout",i.bindGroupLayouts[1]),this.bindGroupSystem.registerLayout("viewport_adaptive_cache",i.bindGroupLayouts[2])}setupUniforms(){const e=Et(this.canvas);e.find(t=>t.name==="cameraPosition").defaultValue=Array.from(this.camera.position),e.find(t=>t.name==="cameraMatrix").defaultValue=this.camera.viewMatrix,e.find(t=>t.name==="viewMatrix").defaultValue=this.camera.viewMatrix,e.find(t=>t.name==="projectionMatrix").defaultValue=this.camera.projectionMatrix,this.viewportBuffer=this.uniformManager.registerBuffer("viewport",e,te.PER_FRAME)}updateUniforms(){const{updates:e,hasNavigation:t}=At(this);t&&(this.camera.updateFromNavigation(this.navigation),this.camera.update(this.canvas.width,this.canvas.height),this.navigation.x=0,this.navigation.y=0,this.navigation.zoom=0,this.navigation.panX=0,this.navigation.panY=0),Object.keys(e).length>0&&this.uniformManager.batchUpdate("viewport",e)}updateBrushPreview(e,t,i,s){this.showBrushPreview=!0,this.brushPreviewPosition=e,this.brushPreviewRadius=t,this.brushPreviewNormal=i,this.brushPreviewDistance=s,this.isDirty=!0}hideBrushPreview(){this.showBrushPreview=!1,this.isDirty=!0}cancelCameraNavigation(){this.navigation.x=0,this.navigation.y=0,this.navigation.zoom=0,this.navigation.panX=0,this.navigation.panY=0}async readGBufferAtPixel(e,t){if(!this.gBufferManager)return null;const i=await this.gBufferManager.readPixel(e,t);return i&&(i.hit=i.distance<d.camera.farPlane-.1),i}getRayFromViewport(e,t){const i=this.canvas.width/this.canvas.height;this.camera.update(this.canvas.width,this.canvas.height);const s=this.camera.viewMatrix,a=[s[0],s[4],s[8]],n=[s[1],s[5],s[9]],o=[-s[2],-s[6],-s[10]],l=d.camera;if(!l.sensorWidth||!l.focalLength)throw new Error("[Viewport] Camera sensor parameters not configured!");const c=l.sensorHeight,p=2*Math.atan(c/(2*l.focalLength)),u=Math.tan(p/2),g=M();return xe(g,a[0]*e*i*u+n[0]*t*u+o[0],a[1]*e*i*u+n[1]*t*u+o[1],a[2]*e*i*u+n[2]*t*u+o[2]),oe(g,g),{origin:[...this.camera.position],direction:Array.from(g)}}async render(){L.emit(k.RENDER_START),this.prepareFrame();const e=this.device.createCommandEncoder(),t=this.webgpu.getCurrentTexture();this.renderMainPass(e,t),this.device.queue.submit([e.finish()]),L.emit(k.RENDER_COMPLETE)}prepareFrame(){this.updateUniforms(),this.uniformManager.updateBuffers(this.device.queue),this.camera.update(this.canvas.width,this.canvas.height)}renderMainPass(e,t){const i=e.beginRenderPass({colorAttachments:[{view:t.createView(),clearValue:{r:.18,g:.18,b:.18,a:1},loadOp:"clear",storeOp:"store"}]});i.setPipeline(this.pipeline);const s=this.bindGroupSystem.getBindGroup("viewport_uniforms","viewport_uniforms_layout"),a=this.bindGroupSystem.getBindGroup("viewport_voxel_hashmap","viewport_voxel_hashmap_layout");if(i.setBindGroup(0,s),i.setBindGroup(1,a),this.bindGroupSystem.definitions.has("viewport_adaptive_cache")){const n=this.bindGroupSystem.getBindGroup("viewport_adaptive_cache","viewport_adaptive_cache");i.setBindGroup(2,n)}else{i.end();return}if(i.draw(3),this.showBrushPreview&&this.brushPreviewPipeline){this.updateBrushPreviewUniforms(),i.setPipeline(this.brushPreviewPipeline);const n=this.bindGroupSystem.getBindGroup("brush_preview","brush_preview_layout");i.setBindGroup(0,n),i.draw(6)}i.end()}updateBrushPreviewUniforms(){const e=this.brushPreviewFloatView;e.set(this.camera.viewMatrix,0),e.set(this.camera.projectionMatrix,16),e[32]=this.brushPreviewPosition[0],e[33]=this.brushPreviewPosition[1],e[34]=this.brushPreviewPosition[2],e[35]=this.brushPreviewRadius,e[36]=this.brushPreviewNormal[0],e[37]=this.brushPreviewNormal[1],e[38]=this.brushPreviewNormal[2],e[39]=this.showBrushPreview?1:0,e[40]=this.camera.position[0],e[41]=this.camera.position[1],e[42]=this.camera.position[2],this.device.queue.writeBuffer(this.brushPreviewUniformBuffer,0,this.brushPreviewUniformData)}createBrushPreviewPipeline(){const{pipeline:e,bindGroupLayout:t}=vt(this.device,this.presentationFormat);this.brushPreviewPipeline=e,this.brushPreviewUniformBuffer=this.resourceManager.createBuffer({label:"Brush Preview Uniform Buffer",size:256,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.bindGroupSystem.defineBindGroup("brush_preview",0,[{binding:0,type:b.BUFFER}]),this.bindGroupSystem.registerLayout("brush_preview_layout",t),this.bindGroupSystem.setResource("brush_preview",0,this.brushPreviewUniformBuffer,b.BUFFER)}createGBufferTextures(){this.gBufferManager&&this.gBufferManager.resize(this.canvas.width,this.canvas.height)}setupInputHandlers(){this.input.onAction("camera:rotate",e=>{this.navigation.x=e.mouse.deltaX*.01,this.navigation.y=-e.mouse.deltaY*.01,this.isDirty=!0}),this.input.onAction("camera:pan",e=>{const t=this.camera.view.distance*.001;this.navigation.panX=-e.mouse.deltaX*t,this.navigation.panY=e.mouse.deltaY*t,this.isDirty=!0}),this.input.onAction("camera:zoom",e=>{this.navigation.zoom=e.delta*.01,this.isDirty=!0}),this.input.onAction("camera:reset",()=>{this.camera.reset(),this.navigation={x:0,y:0,zoom:0,panX:0,panY:0},this.isDirty=!0})}setupWindowEvents(){window.addEventListener("resize",()=>{const t=window.devicePixelRatio,i=Math.max(1,Math.floor(this.canvas.clientWidth*t)),s=Math.max(1,Math.floor(this.canvas.clientHeight*t));this.canvas.width===i&&this.canvas.height===s||(this.canvas.width=i,this.canvas.height=s,this.createGBufferTextures(),this.camera.isDirty=!0,this.camera.update(this.canvas.width,this.canvas.height),this.isDirty=!0,L.emit(k.RESIZE,{width:this.canvas.width,height:this.canvas.height}))});const e=window.devicePixelRatio;this.canvas.width=this.canvas.clientWidth*e,this.canvas.height=this.canvas.clientHeight*e,this.createGBufferTextures()}renderLoop(){this.gui.statsBegin(),this.isDirty&&(this.render(),this.isDirty=!1),this.gui.statsEnd(),this.animationFrameId=requestAnimationFrame(()=>this.renderLoop())}showError(e){const t=document.createElement("div");t.style.cssText=`
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(255, 0, 0, 0.9);
            color: white;
            padding: 20px;
            border-radius: 10px;
            font-family: monospace;
            z-index: 1000;
        `,t.textContent=`Error: ${e}`,document.body.appendChild(t)}invalidateBindGroupCaches(){this.bindGroupSystem.invalidateAll(),this.bindGroupSystem.invalidate("final_composite")}destroy(){this.animationFrameId&&(cancelAnimationFrame(this.animationFrameId),this.animationFrameId=null),this.resizeTimeout&&(clearTimeout(this.resizeTimeout),this.resizeTimeout=null),this.gBufferManager&&(this.gBufferManager.destroy(),this.gBufferManager=null),this.gui&&(this.gui.destroy(),this.gui=null),this.pipelineManager&&(this.pipelineManager.destroy(),this.pipelineManager=null),this.uniformManager&&(this.uniformManager.destroy(),this.uniformManager=null),this.resourceManager&&(this.resourceManager.destroy(),this.resourceManager=null),this.sculptingSystem&&(this.sculptingSystem.destroy&&this.sculptingSystem.destroy(),this.sculptingSystem=null),this.input&&(this.input.destroy(),this.input=null),this.webgpu&&(this.webgpu.destroy(),this.webgpu=null)}}export{Ri as Viewport};
