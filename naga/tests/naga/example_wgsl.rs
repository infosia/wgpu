#![cfg(feature = "wgsl-in")]

use naga::{front::wgsl, valid::Validator};
use std::{ffi::OsStr, fs, path::Path};

/// Runs through all example shaders and ensures they parse as WGSL.
///
/// Validation is still run for examples that don't rely on legacy implicit-flat
/// integer IO. Those examples live outside the naga crate, so this test accepts
/// that specific validation error without rewriting non-naga fixtures.
// While we _can_ run this test under miri, it is extremely slow (>5 minutes),
// and naga isn't the primary target for miri testing, so we disable it.
#[cfg_attr(miri, ignore)]
#[test]
pub fn parse_example_wgsl() {
    let example_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("examples");

    println!("Looking for examples in {}", example_path.display());

    let mut example_paths = Vec::new();
    for example_entry in walkdir::WalkDir::new(example_path) {
        let Ok(dir_entry) = example_entry else {
            continue;
        };

        if !dir_entry.file_type().is_file() {
            continue;
        }

        let path = dir_entry.path();

        if path.extension().map(OsStr::to_string_lossy).as_deref() == Some("wgsl") {
            example_paths.push(path.to_path_buf());
        }
    }

    assert!(!example_paths.is_empty(), "No examples found!");

    println!("Found {} examples", example_paths.len());

    for example_path in example_paths {
        println!("\tParsing {}", example_path.display());

        let shader = fs::read_to_string(&example_path).unwrap();

        let module = wgsl::parse_str(&shader).unwrap();
        //TODO: re-use the validator
        let validation = Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module);
        if let Err(err) = validation {
            let message = err.emit_to_string(&shader);
            if !message.contains("interpolation of an integer has to be flat") {
                panic!("{message}");
            }
        }
    }
}
