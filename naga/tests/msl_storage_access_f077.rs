//! F-077 regression: storage-texture access in MSL type formatting.
//!
//! Several MSL-writer paths format a storage-texture type with EMPTY
//! usage-based access flags (e.g. helper-function parameter types, and
//! globals declared but unused by the entry point). Upstream naga hits
//! `unreachable!("module is not valid")` there; the yawgpu fork falls back
//! to the type's declared access (and only errors when both are empty).

#![cfg(all(feature = "wgsl-in", feature = "msl-out"))]

fn write_msl(src: &str) -> String {
    let module = naga::front::wgsl::parse_str(src).expect("WGSL parse failed");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&module)
    .expect("module validation failed");
    let (msl, _) = naga::back::msl::write_string(
        &module,
        &info,
        &naga::back::msl::Options::default(),
        &naga::back::msl::PipelineOptions::default(),
    )
    .expect("MSL write_string must not panic or error");
    msl
}

/// A read-only storage texture referenced through a helper function: the
/// helper's parameter type is formatted with empty usage access and must
/// fall back to the type's declared `read` access (upstream panics here).
#[test]
fn storage_texture_via_helper_function_does_not_panic() {
    let msl = write_msl(
        r#"
@group(0) @binding(0)
var st: texture_storage_2d<rgba8unorm, read>;
@group(0) @binding(1)
var<storage, read_write> out: vec4<f32>;

fn helper() -> vec4<f32> {
    return textureLoad(st, vec2<i32>(0, 0));
}

@compute @workgroup_size(1)
fn main() {
    out = helper();
}
"#,
    );
    assert!(
        msl.contains("access::read"),
        "emitted MSL should contain 'access::read'; got:\n{msl}"
    );
}

/// A storage texture declared but never used: the writer omits it from the
/// emitted entry point; generation must succeed without panicking.
#[test]
fn unused_storage_texture_does_not_panic() {
    let _ = write_msl(
        r#"
@group(0) @binding(0)
var my_storage: texture_storage_2d<rgba8unorm, read>;

@compute @workgroup_size(1)
fn main() {}
"#,
    );
}
