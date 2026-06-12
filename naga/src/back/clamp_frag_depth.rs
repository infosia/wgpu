/*!
Clamp a fragment shader's written `@builtin(frag_depth)` to the viewport depth range.

WebGPU requires the depth value used by the depth test — including a value written
by the fragment shader via `@builtin(frag_depth)` — to be clamped to the viewport's
`[minDepth, maxDepth]` range. OpenGL and Vulkan perform this clamp automatically, but
Metal (and D3D) do not, so a backend targeting Metal must inject the clamp into the
shader. This transform mirrors Dawn's Tint `ClampFragDepth` pass.

The transform adds a single [`AddressSpace::Immediate`] global holding the
`vec2<f32>(minDepth, maxDepth)` range and wraps every `frag_depth` value produced by
the selected entry point with `clamp(value, range.x, range.y)`. The consumer is
responsible for populating that immediate buffer with the current viewport depth range
(default `[0.0, 1.0]`) for each draw — for the MSL backend, by setting
[`EntryPointResources::immediates_buffer`].

[`AddressSpace::Immediate`]: crate::AddressSpace::Immediate
[`EntryPointResources::immediates_buffer`]: crate::back::msl::EntryPointResources::immediates_buffer
*/

use alloc::{borrow::Cow, string::String};

use thiserror::Error;

use crate::{
    proc::Emitter,
    valid::{Capabilities, ModuleInfo, ValidationError, ValidationFlags, Validator},
    AddressSpace, Arena, Binding, Block, BuiltIn, Expression, Function, GlobalVariable, Handle,
    MathFunction, MemoryDecorations, Module, Scalar, ShaderStage, Span, Statement, SwitchCase,
    Type, TypeInner, VectorSize, WithSpan,
};

/// Error type for [`clamp_frag_depth`].
#[derive(Error, Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub enum ClampFragDepthError {
    /// Re-validating the transformed module failed.
    #[error(transparent)]
    Validation(#[from] WithSpan<ValidationError>),
}

/// How the selected entry point exposes its `frag_depth` output.
#[derive(Clone, Copy)]
enum FragDepthOutput {
    /// The function result is a bare `@builtin(frag_depth) f32`.
    Scalar,
    /// The function result is a struct; `depth_index` selects the `frag_depth` member.
    Struct {
        ty: Handle<Type>,
        member_count: u32,
        depth_index: u32,
    },
}

/// Clamp the `frag_depth` output of `entry_point` to a viewport depth range.
///
/// `module` must be valid. If the named entry point does not write
/// `@builtin(frag_depth)`, this is a no-op and borrowed references are returned.
/// Otherwise it clones `module`, injects an [`AddressSpace::Immediate`]
/// `vec2<f32>` global for the `[minDepth, maxDepth]` range, wraps each returned
/// depth value with `clamp(depth, range.x, range.y)`, re-validates, and returns
/// the transformed module and its fresh [`ModuleInfo`].
///
/// [`AddressSpace::Immediate`]: crate::AddressSpace::Immediate
pub fn clamp_frag_depth<'a>(
    module: &'a Module,
    module_info: &'a ModuleInfo,
    entry_point: (ShaderStage, &str),
) -> Result<(Cow<'a, Module>, Cow<'a, ModuleInfo>), ClampFragDepthError> {
    let (stage, name) = entry_point;
    let output = module.entry_points.iter().find_map(|ep| {
        (ep.stage == stage && ep.name == name)
            .then(|| frag_depth_output(module, &ep.function))
            .flatten()
    });
    let Some(output) = output else {
        return Ok((Cow::Borrowed(module), Cow::Borrowed(module_info)));
    };

    let mut module = module.clone();

    // The viewport depth range, supplied per-draw by the consumer.
    let range_ty = module.types.insert(
        Type {
            name: Some(String::from("naga_frag_depth_clamp_range")),
            inner: TypeInner::Vector {
                size: VectorSize::Bi,
                scalar: Scalar::F32,
            },
        },
        Span::default(),
    );
    let range_global = module.global_variables.append(
        GlobalVariable {
            name: Some(String::from("naga_frag_depth_clamp")),
            space: AddressSpace::Immediate,
            binding: None,
            ty: range_ty,
            init: None,
            memory_decorations: MemoryDecorations::empty(),
        },
        Span::default(),
    );

    for ep in module.entry_points.iter_mut() {
        if ep.stage == stage && ep.name == name {
            let range_expr = ep
                .function
                .expressions
                .append(Expression::GlobalVariable(range_global), Span::default());
            let body = core::mem::take(&mut ep.function.body);
            let mut rewrite = Rewrite {
                expressions: &mut ep.function.expressions,
                range_expr,
                output,
            };
            ep.function.body = rewrite_block(body, &mut rewrite);
            break;
        }
    }

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let module_info = validator.validate(&module)?;
    Ok((Cow::Owned(module), Cow::Owned(module_info)))
}

/// Returns how `function` exposes `frag_depth`, or `None` if it does not write it.
fn frag_depth_output(module: &Module, function: &Function) -> Option<FragDepthOutput> {
    let result = function.result.as_ref()?;
    if matches!(result.binding, Some(Binding::BuiltIn(BuiltIn::FragDepth))) {
        return Some(FragDepthOutput::Scalar);
    }
    if let TypeInner::Struct { ref members, .. } = module.types[result.ty].inner {
        let depth_index = members.iter().position(|member| {
            matches!(member.binding, Some(Binding::BuiltIn(BuiltIn::FragDepth)))
        })?;
        return Some(FragDepthOutput::Struct {
            ty: result.ty,
            member_count: members.len() as u32,
            depth_index: depth_index as u32,
        });
    }
    None
}

/// Mutable state threaded through the recursive body rewrite.
struct Rewrite<'a> {
    expressions: &'a mut Arena<Expression>,
    range_expr: Handle<Expression>,
    output: FragDepthOutput,
}

impl Rewrite<'_> {
    /// Appends the expressions that clamp `value` and returns the new output handle.
    ///
    /// All appended expressions are runtime (emitted) expressions; the caller wraps
    /// them in a [`Statement::Emit`] range placed before the rewritten `Return`.
    fn clamped_output(&mut self, value: Handle<Expression>) -> Handle<Expression> {
        let span = Span::default();
        let range = self.expressions.append(
            Expression::Load {
                pointer: self.range_expr,
            },
            span,
        );
        let min = self.expressions.append(
            Expression::AccessIndex {
                base: range,
                index: 0,
            },
            span,
        );
        let max = self.expressions.append(
            Expression::AccessIndex {
                base: range,
                index: 1,
            },
            span,
        );
        match self.output {
            FragDepthOutput::Scalar => self.clamp(value, min, max, span),
            FragDepthOutput::Struct {
                ty,
                member_count,
                depth_index,
            } => {
                let depth = self.expressions.append(
                    Expression::AccessIndex {
                        base: value,
                        index: depth_index,
                    },
                    span,
                );
                let clamped = self.clamp(depth, min, max, span);
                let mut components = alloc::vec::Vec::with_capacity(member_count as usize);
                for index in 0..member_count {
                    if index == depth_index {
                        components.push(clamped);
                    } else {
                        components.push(
                            self.expressions
                                .append(Expression::AccessIndex { base: value, index }, span),
                        );
                    }
                }
                self.expressions
                    .append(Expression::Compose { ty, components }, span)
            }
        }
    }

    /// Appends a `clamp(value, min, max)` expression.
    fn clamp(
        &mut self,
        value: Handle<Expression>,
        min: Handle<Expression>,
        max: Handle<Expression>,
        span: Span,
    ) -> Handle<Expression> {
        self.expressions.append(
            Expression::Math {
                fun: MathFunction::Clamp,
                arg: value,
                arg1: Some(min),
                arg2: Some(max),
                arg3: None,
            },
            span,
        )
    }
}

/// Rebuilds `block`, clamping every `frag_depth`-bearing `Return` it contains.
///
/// Recurses into nested control flow. `Loop::continuing` cannot contain `Return`,
/// so descending into it is harmless.
fn rewrite_block(block: Block, rewrite: &mut Rewrite<'_>) -> Block {
    let mut out = Block::with_capacity(block.len());
    for (statement, span) in block.span_into_iter() {
        match statement {
            Statement::Return { value: Some(value) } => {
                let mut emitter = Emitter::default();
                emitter.start(rewrite.expressions);
                let clamped = rewrite.clamped_output(value);
                out.extend(emitter.finish(rewrite.expressions));
                out.push(
                    Statement::Return {
                        value: Some(clamped),
                    },
                    span,
                );
            }
            Statement::Block(inner) => {
                out.push(Statement::Block(rewrite_block(inner, rewrite)), span)
            }
            Statement::If {
                condition,
                accept,
                reject,
            } => out.push(
                Statement::If {
                    condition,
                    accept: rewrite_block(accept, rewrite),
                    reject: rewrite_block(reject, rewrite),
                },
                span,
            ),
            Statement::Switch { selector, cases } => {
                let cases = cases
                    .into_iter()
                    .map(|case| SwitchCase {
                        value: case.value,
                        fall_through: case.fall_through,
                        body: rewrite_block(case.body, rewrite),
                    })
                    .collect();
                out.push(Statement::Switch { selector, cases }, span);
            }
            Statement::Loop {
                body,
                continuing,
                break_if,
            } => out.push(
                Statement::Loop {
                    body: rewrite_block(body, rewrite),
                    continuing,
                    break_if,
                },
                span,
            ),
            other => out.push(other, span),
        }
    }
    out
}

#[cfg(all(test, feature = "wgsl-in"))]
mod tests {
    use super::clamp_frag_depth;
    use crate::{
        valid::{Capabilities, ModuleInfo, ValidationFlags, Validator},
        AddressSpace, Expression, Function, MathFunction, Module, ShaderStage,
    };

    fn validate(module: &Module) -> ModuleInfo {
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(module)
            .expect("module should validate")
    }

    fn frag_function<'a>(module: &'a Module, name: &str) -> &'a Function {
        &module
            .entry_points
            .iter()
            .find(|ep| ep.name == name && ep.stage == ShaderStage::Fragment)
            .expect("fragment entry point should exist")
            .function
    }

    fn has_clamp(function: &Function) -> bool {
        function.expressions.iter().any(|(_, expr)| {
            matches!(
                expr,
                Expression::Math {
                    fun: MathFunction::Clamp,
                    ..
                }
            )
        })
    }

    #[test]
    fn clamps_scalar_frag_depth() {
        let module = crate::front::wgsl::parse_str(
            "@fragment fn fs() -> @builtin(frag_depth) f32 { return 2.0; }",
        )
        .expect("wgsl should parse");
        let info = validate(&module);
        assert!(!has_clamp(frag_function(&module, "fs")));

        let (out, _) = clamp_frag_depth(&module, &info, (ShaderStage::Fragment, "fs"))
            .expect("transform should succeed");
        let out = out.into_owned();
        assert!(has_clamp(frag_function(&out, "fs")));
        assert!(out
            .global_variables
            .iter()
            .any(|(_, gv)| gv.space == AddressSpace::Immediate));
    }

    #[test]
    fn clamps_struct_frag_depth() {
        let module = crate::front::wgsl::parse_str(
            "struct Out { @location(0) color: vec4<f32>, @builtin(frag_depth) depth: f32 }
             @fragment fn fs() -> Out { return Out(vec4<f32>(1.0), 2.0); }",
        )
        .expect("wgsl should parse");
        let info = validate(&module);
        let (out, _) = clamp_frag_depth(&module, &info, (ShaderStage::Fragment, "fs"))
            .expect("transform should succeed");
        assert!(has_clamp(frag_function(&out, "fs")));
    }

    #[test]
    fn no_frag_depth_output_is_noop() {
        let module = crate::front::wgsl::parse_str(
            "@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
        )
        .expect("wgsl should parse");
        let info = validate(&module);
        let (out, _) = clamp_frag_depth(&module, &info, (ShaderStage::Fragment, "fs"))
            .expect("transform should succeed");
        assert!(matches!(out, alloc::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn emits_metal_clamp_with_immediate_buffer() {
        let module = crate::front::wgsl::parse_str(
            "@fragment fn fs() -> @builtin(frag_depth) f32 { return 2.0; }",
        )
        .expect("wgsl should parse");
        let info = validate(&module);
        let (out, out_info) = clamp_frag_depth(&module, &info, (ShaderStage::Fragment, "fs"))
            .expect("transform should succeed");

        let mut per_entry_point_map = crate::back::msl::EntryPointResourceMap::default();
        per_entry_point_map.insert(
            alloc::string::String::from("fs"),
            crate::back::msl::EntryPointResources {
                immediates_buffer: Some(30),
                ..Default::default()
            },
        );
        let options = crate::back::msl::Options {
            per_entry_point_map,
            fake_missing_bindings: true,
            ..Default::default()
        };
        let pipeline_options = crate::back::msl::PipelineOptions {
            entry_point: Some((ShaderStage::Fragment, alloc::string::String::from("fs"))),
            ..Default::default()
        };
        let (source, _) =
            crate::back::msl::write_string(&out, &out_info, &options, &pipeline_options)
                .expect("msl generation should succeed");
        assert!(source.contains("metal::clamp"));
        assert!(source.contains("buffer(30)"));
    }
}
