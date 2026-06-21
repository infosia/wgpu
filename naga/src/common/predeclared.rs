//! Generating names for predeclared types.

use crate::{arena::UniqueArena, ir, proc::Alignment, span::Span};

use alloc::format;
use alloc::{
    string::{String, ToString},
    vec,
};

impl ir::PredeclaredType {
    pub fn struct_name(&self) -> String {
        use crate::PredeclaredType as Pt;
        match *self {
            Pt::AtomicCompareExchangeWeakResult(scalar) => {
                format!(
                    "__atomic_compare_exchange_result<{:?},{}>",
                    scalar.kind, scalar.width,
                )
            }
            Pt::ModfResult { size, scalar } => frexp_mod_name("modf", size, scalar),
            Pt::FrexpResult { size, scalar } => frexp_mod_name("frexp", size, scalar),
        }
    }
}

fn frexp_mod_name(function: &str, size: Option<ir::VectorSize>, scalar: ir::Scalar) -> String {
    let bits = 8 * scalar.width;
    match size {
        Some(size) => {
            let size = size as u8;
            format!("__{function}_result_vec{size}_f{bits}")
        }
        None => format!("__{function}_result_f{bits}"),
    }
}

/// Build the predeclared result struct for `frexp`.
///
/// The constant evaluator uses this too, so const-evaluated `frexp` expressions
/// compose exactly the same struct type that the frontend predeclares.
pub(crate) fn frexp_result_type(
    types: &mut UniqueArena<ir::Type>,
    name: String,
    size: Option<ir::VectorSize>,
    scalar: ir::Scalar,
) -> ir::Type {
    let float_ty = types.insert(
        ir::Type {
            name: None,
            inner: ir::TypeInner::Scalar(scalar),
        },
        Span::UNDEFINED,
    );

    let int_scalar = ir::Scalar {
        kind: ir::ScalarKind::Sint,
        width: scalar.width,
    };
    let int_ty = types.insert(
        ir::Type {
            name: None,
            inner: ir::TypeInner::Scalar(int_scalar),
        },
        Span::UNDEFINED,
    );

    let (fract_member_ty, exp_member_ty, fract_size, exp_size, exp_alignment) =
        if let Some(size) = size {
            let vec_float_ty = types.insert(
                ir::Type {
                    name: None,
                    inner: ir::TypeInner::Vector { size, scalar },
                },
                Span::UNDEFINED,
            );
            let vec_int_ty = types.insert(
                ir::Type {
                    name: None,
                    inner: ir::TypeInner::Vector {
                        size,
                        scalar: int_scalar,
                    },
                },
                Span::UNDEFINED,
            );
            let fract_size = size as u32 * scalar.width as u32;
            let exp_size = size as u32 * int_scalar.width as u32;
            let exp_alignment = Alignment::from(size) * Alignment::from_width(int_scalar.width);
            (
                vec_float_ty,
                vec_int_ty,
                fract_size,
                exp_size,
                exp_alignment,
            )
        } else {
            (
                float_ty,
                int_ty,
                scalar.width as u32,
                int_scalar.width as u32,
                Alignment::from_width(int_scalar.width),
            )
        };
    let fract_alignment = if let Some(size) = size {
        Alignment::from(size) * Alignment::from_width(scalar.width)
    } else {
        Alignment::from_width(scalar.width)
    };
    let alignment = fract_alignment.max(exp_alignment);
    let second_offset = exp_alignment.round_up(fract_size);

    ir::Type {
        name: Some(name),
        inner: ir::TypeInner::Struct {
            members: vec![
                ir::StructMember {
                    name: Some("fract".to_string()),
                    ty: fract_member_ty,
                    binding: None,
                    offset: 0,
                },
                ir::StructMember {
                    name: Some("exp".to_string()),
                    ty: exp_member_ty,
                    binding: None,
                    offset: second_offset,
                },
            ],
            alignment,
            span: second_offset + exp_size,
        },
    }
}
