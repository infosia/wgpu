/*!
Apply a render pipeline's constant `multisample.mask` to a fragment shader.

WebGPU's `GPUMultisampleState.mask` is AND-ed into the per-sample coverage of
every fragment, together with the rasterization mask and the shader's own
`@builtin(sample_mask)` output. Vulkan and OpenGL expose this as a fixed-function
pipeline sample mask (`VkPipelineMultisampleStateCreateInfo::pSampleMask`), but
Metal has **no** equivalent API — neither `MTLRenderPipelineDescriptor` nor
`MTLRenderCommandEncoder` carries a constant sample mask. A backend targeting
Metal must therefore fold the mask into the shader by AND-ing it into the
fragment entry point's `@builtin(sample_mask)` output (synthesizing one if the
shader does not already write it). This transform mirrors Dawn's Tint pass.

The mask is a pipeline-creation-time constant, so it is baked in as a literal
(no per-draw uniform is needed — contrast [`clamp_frag_depth`], whose viewport
range is dynamic). The transform is a no-op when `mask == u32::MAX` (the WebGPU
default, which leaves coverage unchanged).

Handled fragment-result shapes:

- bare `@builtin(sample_mask) u32` → `value & mask`,
- a struct with a `@builtin(sample_mask)` member → that member `& mask`,
- a struct **without** a sample-mask member → a member is appended (`= mask`),
- any other bare output (`@location`, `@builtin(frag_depth)`) → wrapped in a new
  struct that adds a sample-mask member (`= mask`).

A fragment entry point with **no** result (no color or `frag_depth` output) is a
no-op: there is nothing whose coverage the mask could observably restrict via
this shader-side mechanism.

[`clamp_frag_depth`]: crate::back::clamp_frag_depth::clamp_frag_depth
*/

use alloc::{borrow::Cow, string::String, vec::Vec};

use thiserror::Error;

use crate::{
    proc::{Alignment, Emitter, Layouter},
    valid::{Capabilities, ModuleInfo, ValidationError, ValidationFlags, Validator},
    Arena, BinaryOperator, Binding, Block, BuiltIn, Expression, FunctionResult, Handle, Literal,
    Module, Scalar, ShaderStage, Span, Statement, StructMember, SwitchCase, Type, TypeInner,
    WithSpan,
};

/// Error type for [`apply_sample_mask`].
#[derive(Error, Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub enum ApplySampleMaskError {
    /// Computing the layout of the fragment result type failed.
    #[error("could not compute the fragment output layout")]
    Layout,
    /// Re-validating the transformed module failed.
    #[error(transparent)]
    Validation(#[from] WithSpan<ValidationError>),
}

/// How the selected entry point exposes (or will expose) its `sample_mask` output.
enum SampleMaskOutput {
    /// The result is a bare `@builtin(sample_mask) u32`; AND the mask into it.
    Scalar,
    /// The result is a struct already carrying a `sample_mask` member at `index`;
    /// AND the mask into that member and recompose.
    ExistingMember {
        ty: Handle<Type>,
        member_count: u32,
        index: u32,
    },
    /// The result is a struct without a sample-mask member; recompose into
    /// `new_ty`, appending the mask as the final member.
    AppendMember {
        new_ty: Handle<Type>,
        orig_member_count: u32,
    },
    /// The result is a bare non-sample-mask output; wrap it in `new_ty`, a struct
    /// `{ original, sample_mask }`.
    Wrap { new_ty: Handle<Type> },
}

/// AND the constant pipeline `mask` into the `sample_mask` output of `entry_point`.
///
/// `module` must be valid. Returns borrowed references unchanged when `mask` is
/// `u32::MAX` (the default) or the fragment entry point has no result. Otherwise
/// it clones `module`, ensures the fragment result carries a `@builtin(sample_mask)`
/// member equal to `existing & mask` (or `mask` when the shader writes none),
/// re-validates, and returns the transformed module and its fresh [`ModuleInfo`].
pub fn apply_sample_mask<'a>(
    module: &'a Module,
    module_info: &'a ModuleInfo,
    entry_point: (ShaderStage, &str),
    mask: u32,
) -> Result<(Cow<'a, Module>, Cow<'a, ModuleInfo>), ApplySampleMaskError> {
    if mask == u32::MAX {
        return Ok((Cow::Borrowed(module), Cow::Borrowed(module_info)));
    }
    let (stage, name) = entry_point;
    let Some(ep_index) = module
        .entry_points
        .iter()
        .position(|ep| ep.stage == stage && ep.name == name)
    else {
        return Ok((Cow::Borrowed(module), Cow::Borrowed(module_info)));
    };
    let function = &module.entry_points[ep_index].function;
    let Some(result) = function.result.as_ref() else {
        // No fragment output: nothing for the mask to restrict via the shader.
        return Ok((Cow::Borrowed(module), Cow::Borrowed(module_info)));
    };

    // Classify the existing result. Layout-dependent injection (append/wrap) is
    // computed against the original (still-valid) module.
    let mut layouter = Layouter::default();
    let existing = classify_existing(module, result);

    let mut module = module.clone();
    let u32_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::U32),
        },
        Span::default(),
    );

    let output = match existing {
        Some(Existing::Scalar) => SampleMaskOutput::Scalar,
        Some(Existing::Member {
            ty,
            member_count,
            index,
        }) => SampleMaskOutput::ExistingMember {
            ty,
            member_count,
            index,
        },
        None => {
            layouter
                .update(module.to_ctx())
                .map_err(|_| ApplySampleMaskError::Layout)?;
            build_injected_output(
                &mut module,
                &layouter,
                result.ty,
                result.binding.clone(),
                u32_ty,
            )
        }
    };

    // Point the result at the new struct type when one was synthesized.
    let new_result = match &output {
        &SampleMaskOutput::AppendMember { new_ty, .. } | &SampleMaskOutput::Wrap { new_ty } => {
            Some(FunctionResult {
                ty: new_ty,
                binding: None,
            })
        }
        &SampleMaskOutput::Scalar | &SampleMaskOutput::ExistingMember { .. } => None,
    };

    let ep = &mut module.entry_points[ep_index].function;
    if let Some(new_result) = new_result {
        ep.result = Some(new_result);
    }
    let mask_expr = ep
        .expressions
        .append(Expression::Literal(Literal::U32(mask)), Span::default());
    let body = core::mem::take(&mut ep.body);
    let mut rewrite = Rewrite {
        expressions: &mut ep.expressions,
        mask_expr,
        output,
    };
    ep.body = rewrite_block(body, &mut rewrite);

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let module_info = validator.validate(&module)?;
    Ok((Cow::Owned(module), Cow::Owned(module_info)))
}

/// A classification that does not require layout information.
enum Existing {
    Scalar,
    Member {
        ty: Handle<Type>,
        member_count: u32,
        index: u32,
    },
}

/// Classifies an existing `sample_mask` output, if the result already writes one.
fn classify_existing(module: &Module, result: &FunctionResult) -> Option<Existing> {
    if matches!(result.binding, Some(Binding::BuiltIn(BuiltIn::SampleMask))) {
        return Some(Existing::Scalar);
    }
    if let TypeInner::Struct { ref members, .. } = module.types[result.ty].inner {
        if let Some(index) = members
            .iter()
            .position(|m| matches!(m.binding, Some(Binding::BuiltIn(BuiltIn::SampleMask))))
        {
            return Some(Existing::Member {
                ty: result.ty,
                member_count: members.len() as u32,
                index: index as u32,
            });
        }
    }
    None
}

/// Builds a new struct result that adds a `sample_mask` member, for a result that
/// does not already write one (either a struct of other outputs, or a bare output).
fn build_injected_output(
    module: &mut Module,
    layouter: &Layouter,
    orig_ty: Handle<Type>,
    orig_binding: Option<Binding>,
    u32_ty: Handle<Type>,
) -> SampleMaskOutput {
    let mask_member = |offset: u32| StructMember {
        name: Some(String::from("naga_sample_mask")),
        ty: u32_ty,
        binding: Some(Binding::BuiltIn(BuiltIn::SampleMask)),
        offset,
    };

    if let TypeInner::Struct {
        ref members, span, ..
    } = module.types[orig_ty].inner
    {
        // Append a sample-mask member after the existing struct.
        let orig_member_count = members.len() as u32;
        let mut members = members.clone();
        let alignment = layouter[orig_ty].alignment;
        let offset = Alignment::FOUR.round_up(span);
        members.push(mask_member(offset));
        let new_span = alignment.round_up(offset + 4);
        let new_ty = module.types.insert(
            Type {
                name: Some(String::from("naga_sample_mask_output")),
                inner: TypeInner::Struct {
                    members,
                    alignment,
                    span: new_span,
                },
            },
            Span::default(),
        );
        SampleMaskOutput::AppendMember {
            new_ty,
            orig_member_count,
        }
    } else {
        // Wrap a bare output into a struct `{ original, sample_mask }`.
        let layout = layouter[orig_ty];
        let mask_offset = Alignment::FOUR.round_up(layout.size);
        let alignment = layout.alignment.max(Alignment::FOUR);
        let span = alignment.round_up(mask_offset + 4);
        let members = alloc::vec![
            StructMember {
                name: Some(String::from("naga_output")),
                ty: orig_ty,
                binding: orig_binding,
                offset: 0,
            },
            mask_member(mask_offset),
        ];
        let new_ty = module.types.insert(
            Type {
                name: Some(String::from("naga_sample_mask_output")),
                inner: TypeInner::Struct {
                    members,
                    alignment,
                    span,
                },
            },
            Span::default(),
        );
        SampleMaskOutput::Wrap { new_ty }
    }
}

/// Mutable state threaded through the recursive body rewrite.
struct Rewrite<'a> {
    expressions: &'a mut Arena<Expression>,
    mask_expr: Handle<Expression>,
    output: SampleMaskOutput,
}

impl Rewrite<'_> {
    /// Appends the expressions producing the masked output and returns its handle.
    fn masked_output(&mut self, value: Handle<Expression>) -> Handle<Expression> {
        let span = Span::default();
        match self.output {
            SampleMaskOutput::Scalar => self.and(value, span),
            SampleMaskOutput::ExistingMember {
                ty,
                member_count,
                index,
            } => {
                let current = self
                    .expressions
                    .append(Expression::AccessIndex { base: value, index }, span);
                let masked = self.and(current, span);
                let components = (0..member_count)
                    .map(|i| {
                        if i == index {
                            masked
                        } else {
                            self.expressions.append(
                                Expression::AccessIndex {
                                    base: value,
                                    index: i,
                                },
                                span,
                            )
                        }
                    })
                    .collect();
                self.expressions
                    .append(Expression::Compose { ty, components }, span)
            }
            SampleMaskOutput::AppendMember {
                new_ty,
                orig_member_count,
            } => {
                let mut components: Vec<_> = (0..orig_member_count)
                    .map(|i| {
                        self.expressions.append(
                            Expression::AccessIndex {
                                base: value,
                                index: i,
                            },
                            span,
                        )
                    })
                    .collect();
                components.push(self.mask_expr);
                self.expressions.append(
                    Expression::Compose {
                        ty: new_ty,
                        components,
                    },
                    span,
                )
            }
            SampleMaskOutput::Wrap { new_ty } => self.expressions.append(
                Expression::Compose {
                    ty: new_ty,
                    components: alloc::vec![value, self.mask_expr],
                },
                span,
            ),
        }
    }

    /// Appends `value & mask`.
    fn and(&mut self, value: Handle<Expression>, span: Span) -> Handle<Expression> {
        self.expressions.append(
            Expression::Binary {
                op: BinaryOperator::And,
                left: value,
                right: self.mask_expr,
            },
            span,
        )
    }
}

/// Rebuilds `block`, masking every result-bearing `Return` it contains.
fn rewrite_block(block: Block, rewrite: &mut Rewrite<'_>) -> Block {
    let mut out = Block::with_capacity(block.len());
    for (statement, span) in block.span_into_iter() {
        match statement {
            Statement::Return { value: Some(value) } => {
                let mut emitter = Emitter::default();
                emitter.start(rewrite.expressions);
                let masked = rewrite.masked_output(value);
                out.extend(emitter.finish(rewrite.expressions));
                out.push(
                    Statement::Return {
                        value: Some(masked),
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
    use super::apply_sample_mask;
    use crate::{
        valid::{Capabilities, ModuleInfo, ValidationFlags, Validator},
        BinaryOperator, Expression, Function, Module, ShaderStage,
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

    fn has_and(function: &Function) -> bool {
        function.expressions.iter().any(|(_, expr)| {
            matches!(
                expr,
                Expression::Binary {
                    op: BinaryOperator::And,
                    ..
                }
            )
        })
    }

    #[test]
    fn default_mask_is_noop() {
        let module = crate::front::wgsl::parse_str(
            "@fragment fn fs() -> @builtin(sample_mask) u32 { return 0xffffffffu; }",
        )
        .expect("wgsl should parse");
        let info = validate(&module);
        let (out, _) = apply_sample_mask(&module, &info, (ShaderStage::Fragment, "fs"), u32::MAX)
            .expect("transform should succeed");
        assert!(matches!(out, alloc::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn ands_scalar_sample_mask() {
        let module = crate::front::wgsl::parse_str(
            "@fragment fn fs() -> @builtin(sample_mask) u32 { return 5u; }",
        )
        .expect("wgsl should parse");
        let info = validate(&module);
        assert!(!has_and(frag_function(&module, "fs")));
        let (out, _) = apply_sample_mask(&module, &info, (ShaderStage::Fragment, "fs"), 0b0101)
            .expect("transform should succeed");
        assert!(has_and(frag_function(&out, "fs")));
    }

    #[test]
    fn ands_struct_member_sample_mask() {
        let module = crate::front::wgsl::parse_str(
            "struct Out { @builtin(sample_mask) mask: u32, @location(0) color: vec4<f32> }
             @fragment fn fs() -> Out { return Out(15u, vec4<f32>(1.0)); }",
        )
        .expect("wgsl should parse");
        let info = validate(&module);
        let (out, _) = apply_sample_mask(&module, &info, (ShaderStage::Fragment, "fs"), 0b0101)
            .expect("transform should succeed");
        assert!(has_and(frag_function(&out, "fs")));
    }

    #[test]
    fn appends_member_when_struct_lacks_sample_mask() {
        let module = crate::front::wgsl::parse_str(
            "struct Out { @location(0) color: vec4<f32> }
             @fragment fn fs() -> Out { return Out(vec4<f32>(1.0)); }",
        )
        .expect("wgsl should parse");
        let info = validate(&module);
        let (out, out_info) =
            apply_sample_mask(&module, &info, (ShaderStage::Fragment, "fs"), 0b0101)
                .expect("transform should succeed");
        // Re-validates internally; just confirm it owns a transformed module.
        assert!(matches!(out, alloc::borrow::Cow::Owned(_)));
        let _ = out_info;
    }

    #[test]
    fn wraps_bare_location_output() {
        let module = crate::front::wgsl::parse_str(
            "@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
        )
        .expect("wgsl should parse");
        let info = validate(&module);
        let (out, _) = apply_sample_mask(&module, &info, (ShaderStage::Fragment, "fs"), 0b0101)
            .expect("transform should succeed");
        assert!(matches!(out, alloc::borrow::Cow::Owned(_)));
    }

    #[test]
    fn no_result_is_noop() {
        let module =
            crate::front::wgsl::parse_str("@fragment fn fs() { }").expect("wgsl should parse");
        let info = validate(&module);
        let (out, _) = apply_sample_mask(&module, &info, (ShaderStage::Fragment, "fs"), 0b0101)
            .expect("transform should succeed");
        assert!(matches!(out, alloc::borrow::Cow::Borrowed(_)));
    }

    #[cfg(feature = "msl-out")]
    #[test]
    fn emits_metal_sample_mask_and_for_bare_output() {
        // A bare @location output gets wrapped into a struct carrying a
        // [[sample_mask]] member set to the constant mask.
        let module = crate::front::wgsl::parse_str(
            "@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
        )
        .expect("wgsl should parse");
        let info = validate(&module);
        let (out, out_info) =
            apply_sample_mask(&module, &info, (ShaderStage::Fragment, "fs"), 0b0101)
                .expect("transform should succeed");

        let options = crate::back::msl::Options {
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
        assert!(source.contains("sample_mask"));
    }
}
