use alloc::format;

use super::parse_str;

#[test]
fn parse_comment() {
    parse_str(
        "//
        ////
        ///////////////////////////////////////////////////////// asda
        //////////////////// dad ////////// /
        /////////////////////////////////////////////////////////////////////////////////////////////////////
        //
    ",
    )
    .unwrap();
}

#[test]
fn parse_types() {
    parse_str("const a : i32 = 2;").unwrap();
    parse_str("const a : u64 = 2lu;").unwrap();
    assert!(parse_str("const a : x32 = 2;").is_err());
    parse_str("var t: texture_2d<f32>;").unwrap();
    parse_str("var t: texture_cube_array<i32>;").unwrap();
    parse_str("var t: texture_multisampled_2d<u32>;").unwrap();
    parse_str("var t: texture_storage_1d<rgba8uint,write>;").unwrap();
    parse_str("var t: texture_storage_3d<r32float,read>;").unwrap();
}

#[test]
fn parse_type_inference() {
    parse_str(
        "
        fn foo() {
            let a = 2u;
            let b: u32 = a;
            var x = 3.;
            var y = vec2<f32>(1, 2);
        }",
    )
    .unwrap();
    assert!(parse_str(
        "
        fn foo() { let c : i32 = 2.0; }",
    )
    .is_err());
}

#[test]
fn parse_type_cast() {
    parse_str(
        "
        const a : i32 = 2;
        fn main() {
            var x: f32 = f32(a);
            x = f32(i32(a + 1) / 2);
        }
    ",
    )
    .unwrap();
    parse_str(
        "
        fn main() {
            let x: vec2<f32> = vec2<f32>(1.0, 2.0);
            let y: vec2<u32> = vec2<u32>(x);
        }
    ",
    )
    .unwrap();
    parse_str(
        "
        fn main() {
            let x: vec2<f32> = vec2<f32>(0.0);
        }
    ",
    )
    .unwrap();
    assert!(parse_str(
        "
        fn main() {
            let x: vec2<i32> = vec2<i32>(0.0, 0.0);
        }
    ",
    )
    .is_err());
}

#[test]
fn parse_type_coercion() {
    parse_str(
        "
        fn foo(bar: f32) {}
        fn main() {
            foo(0);
        }
    ",
    )
    .unwrap();
    assert!(parse_str(
        "
        fn foo(bar: i32) {}
        fn main() {
            foo(0.0);
        }
    ",
    )
    .is_err());
}

#[test]
fn parse_struct() {
    parse_str(
        "
        struct Foo { x: i32 }
        struct Bar {
            @size(16) x: vec2<i32>,
            @align(16) y: f32,
            @size(32) @align(128) z: vec3<f32>,
        };
        struct Empty {}
        var<storage,read_write> s: Foo;
    ",
    )
    .unwrap();
}

#[test]
fn parse_standard_fun() {
    parse_str(
        "
        fn main() {
            var x: i32 = min(max(1, 2), 3);
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_statement() {
    parse_str(
        "
        fn main() {
            ;
            {}
            {;}
        }
    ",
    )
    .unwrap();

    parse_str(
        "
        fn foo() {}
        fn bar() { foo(); }
    ",
    )
    .unwrap();
}

#[test]
fn parse_if() {
    parse_str(
        "
        fn main() {
            if true {
                discard;
            } else {}
            if 0 != 1 {}
            if false {
                return;
            } else if true {
                return;
            } else {}
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_parentheses_if() {
    parse_str(
        "
        fn main() {
            if (true) {
                discard;
            } else {}
            if (0 != 1) {}
            if (false) {
                return;
            } else if (true) {
                return;
            } else {}
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_loop() {
    parse_str(
        "
        fn main() {
            var i: i32 = 0;
            loop {
                if i == 1 { break; }
                continuing { i = 1; }
            }
            loop {
                if i == 0 { continue; }
                break;
            }
        }
    ",
    )
    .unwrap();
    parse_str(
        "
        fn main() {
            var found: bool = false;
            var i: i32 = 0;
            while !found {
                if i == 10 {
                    found = true;
                }

                i = i + 1;
            }
        }
    ",
    )
    .unwrap();
    parse_str(
        "
        fn main() {
            while true {
                break;
            }
        }
    ",
    )
    .unwrap();
    parse_str(
        "
        fn main() {
            var a: i32 = 0;
            for(var i: i32 = 0; i < 4; i = i + 1) {
                a = a + 2;
            }
        }
    ",
    )
    .unwrap();
    parse_str(
        "
        fn main() {
            for(;;) {
                break;
            }
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_switch() {
    parse_str(
        "
        fn main() {
            var pos: f32;
            switch (3) {
                case 0, 1: { pos = 0.0; }
                case 2: { pos = 1.0; }
                default: { pos = 3.0; }
            }
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_switch_optional_colon_in_case() {
    parse_str(
        "
        fn main() {
            var pos: f32;
            switch (3) {
                case 0, 1 { pos = 0.0; }
                case 2 { pos = 1.0; }
                default { pos = 3.0; }
            }
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_switch_default_in_case() {
    parse_str(
        "
        fn main() {
            var pos: f32;
            switch (3) {
                case 0, 1: { pos = 0.0; }
                case 2: {}
                case default, 3: { pos = 3.0; }
            }
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_parentheses_switch() {
    parse_str(
        "
        fn main() {
            var pos: i32;
            switch pos + 1 {
                default: { pos = 3; }
            }
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_texture_load() {
    parse_str(
        "
        var t: texture_3d<u32>;
        fn foo() {
            let r: vec4<u32> = textureLoad(t, vec3<u32>(0u, 1u, 2u), 1);
        }
    ",
    )
    .unwrap();
    parse_str(
        "
        var t: texture_2d_array<i32>;
        fn foo() {
            let r: vec4<i32> = textureLoad(t, vec2<i32>(10, 20), 2, 3);
        }
    ",
    )
    .unwrap();
    parse_str(
        "
        var t: texture_storage_1d<r32float,read>;
        fn foo() {
            let r: vec4<f32> = textureLoad(t, 10);
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_texture_store() {
    parse_str(
        "
        var t: texture_storage_2d<rgba8unorm,write>;
        fn foo() {
            textureStore(t, vec2<i32>(10, 20), vec4<f32>(0.0, 1.0, 2.0, 3.0));
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_texture_query() {
    parse_str(
        "
        var t: texture_multisampled_2d<f32>;
        fn foo() {
            let dim = textureDimensions(t);
            let samples = textureNumSamples(t);
        }
    ",
    )
    .unwrap();
    parse_str(
        "
        var t: texture_2d_array<f32>;
        fn foo() {
            let dim = textureDimensions(t);
            let levels = textureNumLevels(t);
            let layers = textureNumLayers(t);
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_postfix() {
    parse_str(
        "fn foo() {
        let x: f32 = vec4<f32>(1.0, 2.0, 3.0, 4.0).xyz.rgbr.aaaa.wz.g;
        let y: f32 = fract(vec2<f32>(0.5, x)).x;
    }",
    )
    .unwrap();

    let err = parse_str(
        "fn foo() {
        let v = mat4x4<f32>().x;
    }",
    )
    .unwrap_err();
    assert_eq!(err.message(), "invalid field accessor `x`");
}

#[test]
fn parse_expressions() {
    parse_str("fn foo() {
        let x: f32 = select(0.0, 1.0, true);
        let y: vec2<f32> = select(vec2<f32>(1.0, 1.0), vec2<f32>(x, x), vec2<bool>((x < 0.5), (x > 0.5)));
        let z: bool = !(0.0 == 1.0);
    }").unwrap();
}

#[test]
fn parse_assignment_statements() {
    parse_str(
        "
        struct Foo { x: i32 };

        fn foo() {
            var x: u32 = 0u;
            x++;
            x--;
            x = 1u;
            x += 1u;
            var v: vec2<f32> = vec2<f32>(1.0, 1.0);
            v[0] += 1.0;
            (v)[0] += 1.0;
            var s: Foo = Foo(0);
            s.x -= 1;
            (s.x) -= 1;
            (s).x -= 1;
            _ = 5u;
    }",
    )
    .unwrap();

    let error = parse_str(
        "fn foo() {
        x|x++;
    }",
    )
    .unwrap_err();
    assert_eq!(
        error.message(),
        "expected assignment or increment/decrement, found \"|\"",
    );
}

#[test]
fn parse_local_var_address_space() {
    parse_str(
        "
        fn foo() {
            var<function> a = true;
            var<function> b: i32 = 5;
            var c = 10;
        }",
    )
    .unwrap();

    let error = parse_str(
        "fn foo() {
            var<private> x: i32 = 5;
        }",
    )
    .unwrap_err();
    assert_eq!(
        error.message(),
        "invalid address space for local variable: `private`",
    );

    let error = parse_str(
        "fn foo() {
            var<storage> x: i32 = 5;
        }",
    )
    .unwrap_err();
    assert_eq!(
        error.message(),
        "invalid address space for local variable: `storage`",
    );
}

#[test]
fn binary_expression_mixed_scalar_and_vector_operands() {
    for (operand, expect_splat) in [
        ('<', false),
        ('>', false),
        ('&', false),
        ('|', false),
        ('+', true),
        ('-', true),
        ('*', false),
        ('/', true),
        ('%', true),
    ] {
        let module = parse_str(&format!(
            "
            @fragment
            fn main(@location(0) some_vec: vec3<f32>) -> @location(0) vec4<f32> {{
                if (all(1.0 {operand} some_vec)) {{
                    return vec4(0.0);
                }}
                return vec4(1.0);
            }}
            "
        ))
        .unwrap();

        let expressions = &&module.entry_points[0].function.expressions;

        let found_expressions = expressions
            .iter()
            .filter(|&(_, e)| {
                if let crate::Expression::Binary { left, .. } = *e {
                    matches!(
                        (expect_splat, &expressions[left]),
                        (false, &crate::Expression::Literal(crate::Literal::F32(..)))
                            | (true, &crate::Expression::Splat { .. })
                    )
                } else {
                    false
                }
            })
            .count();

        assert_eq!(
            found_expressions,
            1,
            "expected `{operand}` expression {} splat",
            if expect_splat { "with" } else { "without" }
        );
    }

    let module = parse_str(
        "@fragment
        fn main(mat: mat3x3<f32>) {
            let vec = vec3<f32>(1.0, 1.0, 1.0);
            let result = mat / vec;
        }",
    )
    .unwrap();
    let expressions = &&module.entry_points[0].function.expressions;
    let found_splat = expressions.iter().any(|(_, e)| {
        if let crate::Expression::Binary { left, .. } = *e {
            matches!(&expressions[left], &crate::Expression::Splat { .. })
        } else {
            false
        }
    });
    assert!(!found_splat, "'mat / vec' should not be splatted");
}

#[test]
fn parse_pointers() {
    parse_str(
        "fn foo(a: ptr<function, f32>) -> f32 { return *a; }
    fn bar() {
        var x: f32 = 1.0;
        let px = &x;
        let py = foo(px);
    }",
    )
    .unwrap();
}

#[test]
fn parse_struct_instantiation() {
    parse_str(
        "
    struct Foo {
        a: f32,
        b: vec3<f32>,
    }

    @fragment
    fn fs_main() {
        var foo: Foo = Foo(0.0, vec3<f32>(0.0, 1.0, 42.0));
    }
    ",
    )
    .unwrap();
}

#[test]
fn parse_array_length() {
    parse_str(
        "
        struct Foo {
            data: array<u32>
        } // this is used as both input and output for convenience

        @group(0) @binding(0)
        var<storage> foo: Foo;

        @group(0) @binding(1)
        var<storage> bar: array<u32>;

        fn baz() {
            var x: u32 = arrayLength(foo.data);
            var y: u32 = arrayLength(bar);
        }
        ",
    )
    .unwrap();
}

#[test]
fn parse_storage_buffers() {
    parse_str(
        "
        @group(0) @binding(0)
        var<storage> foo: array<u32>;
        ",
    )
    .unwrap();
    parse_str(
        "
        @group(0) @binding(0)
        var<storage,read> foo: array<u32>;
        ",
    )
    .unwrap();
    parse_str(
        "
        @group(0) @binding(0)
        var<storage,write> foo: array<u32>;
        ",
    )
    .unwrap();
    parse_str(
        "
        @group(0) @binding(0)
        var<storage,read_write> foo: array<u32>;
        ",
    )
    .unwrap();
}

#[test]
fn parse_alias() {
    parse_str(
        "
        alias Vec4 = vec4<f32>;
        ",
    )
    .unwrap();
}

#[test]
fn shadowing_predeclared_types() {
    parse_str(
        "
        fn test(f32: vec2f) -> vec2f { return f32; }
        ",
    )
    .unwrap();
    parse_str(
        "
        fn test(vec2: vec2f) -> vec2f { return vec2; }
        ",
    )
    .unwrap();
    parse_str(
        "
        alias vec2f = vec2u;
        fn test(v: vec2f) -> vec2u { return v; }
        ",
    )
    .unwrap();
    parse_str(
        "
        struct vec2f { inner: vec2<f32> };
        fn test(v: vec2f) -> vec2<f32> { return v.inner; }
        ",
    )
    .unwrap();
}

#[test]
fn parse_texture_load_store_expecting_four_args() {
    for (func, texture) in [
        (
            "textureStore",
            "texture_storage_2d_array<rg11b10ufloat, write>",
        ),
        ("textureLoad", "texture_2d_array<i32>"),
    ] {
        let error = parse_str(&format!(
            "
            @group(0) @binding(0) var tex_los_res: {texture};
            @compute
            @workgroup_size(1)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                var color = vec4(1, 1, 1, 1);
                {func}(tex_los_res, id, color);
            }}
            "
        ))
        .unwrap_err();
        assert_eq!(
            error.message(),
            "wrong number of arguments: expected 4, found 3"
        );
    }
}

#[test]
fn parse_repeated_attributes() {
    use crate::{
        front::wgsl::{error::Error, Frontend},
        Span,
    };

    let template_vs = "@vertex fn vs() -> __REPLACE__ vec4<f32> { return vec4<f32>(0.0); }";
    let template_struct = "struct A { __REPLACE__ data: vec3<f32> }";
    let template_resource = "__REPLACE__ var tex_los_res: texture_2d_array<i32>;";
    let template_stage = "__REPLACE__ fn vs() -> vec4<f32> { return vec4<f32>(0.0); }";
    for (attribute, template) in [
        ("align(16)", template_struct),
        ("binding(0)", template_resource),
        ("builtin(position)", template_vs),
        ("compute", template_stage),
        ("fragment", template_stage),
        ("group(0)", template_resource),
        ("interpolate(flat)", template_vs),
        ("invariant", template_vs),
        ("location(0)", template_vs),
        ("size(16)", template_struct),
        ("vertex", template_stage),
        ("early_depth_test(less_equal)", template_resource),
        ("workgroup_size(1)", template_stage),
    ] {
        let shader = template.replace("__REPLACE__", &format!("@{attribute} @{attribute}"));
        let name_length = attribute.rfind('(').unwrap_or(attribute.len()) as u32;
        let span_start = shader.rfind(attribute).unwrap() as u32;
        let span_end = span_start + name_length;
        let expected_span = Span::new(span_start, span_end);

        let result = Frontend::new().inner(&shader);
        assert!(matches!(
            *result.unwrap_err(),
            Error::RepeatedAttribute(span) if span == expected_span
        ));
    }
}

#[test]
fn parse_missing_workgroup_size() {
    use crate::{
        front::wgsl::{error::Error, Frontend},
        Span,
    };

    let shader = "@compute fn vs() -> vec4<f32> { return vec4<f32>(0.0); }";
    let result = Frontend::new().inner(shader);
    assert!(matches!(
        *result.unwrap_err(),
        Error::MissingWorkgroupSize(span) if span == Span::new(1, 8)
    ));
}

mod diagnostic_filter {
    use crate::front::wgsl::assert_parse_err;

    #[test]
    fn intended_global_directive() {
        let shader = "@diagnostic(off, my.lint);";
        assert_parse_err(
            shader,
            "\
error: `@diagnostic(…)` attribute(s) on semicolons are not supported
  ┌─ wgsl:1:1
  │
1 │ @diagnostic(off, my.lint);
  │ ^^^^^^^^^^^^^^^^^^^^^^^^^
  │
  = note: `@diagnostic(…)` attributes are only permitted on `fn`s, some statements, and `switch`/`loop` bodies.
  = note: If you meant to declare a diagnostic filter that applies to the entire module, move this line to the top of the file and remove the `@` symbol.

"
        );
    }

    mod parse_sites_not_yet_supported {
        use crate::front::wgsl::assert_parse_err;

        #[test]
        fn user_rules() {
            let shader = "
fn myfunc() {
    if (true) @diagnostic(off, my.lint) {
        //    ^^^^^^^^^^^^^^^^^^^^^^^^^ not yet supported, should report an error
    }
}
";
            assert_parse_err(shader, "\
error: `@diagnostic(…)` attribute(s) not yet implemented
  ┌─ wgsl:3:15
  │
3 │     if (true) @diagnostic(off, my.lint) {
  │               ^^^^^^^^^^^^^^^^^^^^^^^^^ can't use this on compound statements (yet)
  │
  = note: Let Naga maintainers know that you ran into this at <https://github.com/gfx-rs/wgpu/issues/5320>, so they can prioritize it!

");
        }

        #[test]
        fn unknown_rules() {
            let shader = "
fn myfunc() {
	if (true) @diagnostic(off, wat_is_this) {
		//    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ should emit a warning
	}
}
";
            assert_parse_err(shader, "\
error: `@diagnostic(…)` attribute(s) not yet implemented
  ┌─ wgsl:3:12
  │
3 │     if (true) @diagnostic(off, wat_is_this) {
  │               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ can't use this on compound statements (yet)
  │
  = note: Let Naga maintainers know that you ran into this at <https://github.com/gfx-rs/wgpu/issues/5320>, so they can prioritize it!

");
        }
    }

    mod directive_conflict {
        use crate::front::wgsl::assert_parse_err;

        #[test]
        fn user_rules() {
            let shader = "
diagnostic(off, my.lint);
diagnostic(warning, my.lint);
";
            assert_parse_err(shader, "\
error: found conflicting `diagnostic(…)` rule(s)
  ┌─ wgsl:2:1
  │
2 │ diagnostic(off, my.lint);
  │ ^^^^^^^^^^^^^^^^^^^^^^^^ first rule
3 │ diagnostic(warning, my.lint);
  │ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ second rule
  │
  = note: Multiple `diagnostic(…)` rules with the same rule name conflict unless they are directives and the severity is the same.
  = note: You should delete the rule you don't want.

");
        }

        #[test]
        fn unknown_rules() {
            let shader = "
diagnostic(off, wat_is_this);
diagnostic(warning, wat_is_this);
";
            assert_parse_err(shader, "\
error: found conflicting `diagnostic(…)` rule(s)
  ┌─ wgsl:2:1
  │
2 │ diagnostic(off, wat_is_this);
  │ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ first rule
3 │ diagnostic(warning, wat_is_this);
  │ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ second rule
  │
  = note: Multiple `diagnostic(…)` rules with the same rule name conflict unless they are directives and the severity is the same.
  = note: You should delete the rule you don't want.

");
        }
    }

    mod attribute_conflict {
        use crate::front::wgsl::assert_parse_err;

        #[test]
        fn user_rules() {
            let shader = "
diagnostic(off, my.lint);
diagnostic(warning, my.lint);
";
            assert_parse_err(shader, "\
error: found conflicting `diagnostic(…)` rule(s)
  ┌─ wgsl:2:1
  │
2 │ diagnostic(off, my.lint);
  │ ^^^^^^^^^^^^^^^^^^^^^^^^ first rule
3 │ diagnostic(warning, my.lint);
  │ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ second rule
  │
  = note: Multiple `diagnostic(…)` rules with the same rule name conflict unless they are directives and the severity is the same.
  = note: You should delete the rule you don't want.

");
        }

        #[test]
        fn unknown_rules() {
            let shader = "
diagnostic(off, wat_is_this);
diagnostic(warning, wat_is_this);
";
            assert_parse_err(shader, "\
error: found conflicting `diagnostic(…)` rule(s)
  ┌─ wgsl:2:1
  │
2 │ diagnostic(off, wat_is_this);
  │ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ first rule
3 │ diagnostic(warning, wat_is_this);
  │ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ second rule
  │
  = note: Multiple `diagnostic(…)` rules with the same rule name conflict unless they are directives and the severity is the same.
  = note: You should delete the rule you don't want.

");
        }
    }
}

mod template {
    use crate::front::wgsl::assert_parse_err;

    #[test]
    fn missing_template_end() {
        assert_parse_err(
            "
fn storage() {}
var<storage
",
            "\
error: expected identifier, found \"<\"
  ┌─ wgsl:3:4
  │
3 │ var<storage
  │    ^ expected identifier

",
        );
    }

    #[test]
    fn enumerant_shadowing() {
        assert_parse_err(
            "
fn storage() {}
var<storage> s: u32;
",
            "\
error: identifier `storage` resolves to a declaration
  ┌─ wgsl:3:5
  │
3 │ var<storage> s: u32;
  │     ^^^^^^^ needs to resolve to a predeclared enumerant

",
        );
    }

    #[test]
    fn unexpected_expr_as_enumerant() {
        assert_parse_err(
            "
var<1 + 1> s: u32;
",
            "\
error: unexpected expression
  ┌─ wgsl:2:5
  │
2 │ var<1 + 1> s: u32;
  │     ^^^^^ needs to be an identifier resolving to a predeclared enumerant

",
        );
    }

    #[test]
    fn unused_exprs_for_template() {
        assert_parse_err(
            "
var<storage, read_write, extra0, extra1> s: u32;
",
            "\
error: unused expressions for template
  ┌─ wgsl:2:26
  │
2 │ var<storage, read_write, extra0, extra1> s: u32;
  │                          ^^^^^^  ^^^^^^ unused
  │                          │\x20\x20\x20\x20\x20\x20\x20\x20
  │                          unused

",
        );
    }

    #[test]
    fn unused_template_list_for_fn() {
        assert_parse_err(
            "
fn inner_test() {}
fn test() {
    inner_test<unused_template_arg>();
}
",
            "\
error: unused expressions for template
  ┌─ wgsl:4:16
  │
4 │     inner_test<unused_template_arg>();
  │                ^^^^^^^^^^^^^^^^^^^ unused

",
        );
    }

    #[test]
    fn unused_template_list_for_struct() {
        assert_parse_err(
            "
struct test_struct {}
fn test() {
    _ = test_struct<unused_template_arg>();
}
",
            "\
error: unused expressions for template
  ┌─ wgsl:4:21
  │
4 │     _ = test_struct<unused_template_arg>();
  │                     ^^^^^^^^^^^^^^^^^^^ unused

",
        );
    }

    #[test]
    fn unused_template_list_for_alias() {
        assert_parse_err(
            "
alias test_alias = f32;
fn test() {
    _ = test_alias<unused_template_arg>();
}
",
            "\
error: unused expressions for template
  ┌─ wgsl:4:20
  │
4 │     _ = test_alias<unused_template_arg>();
  │                    ^^^^^^^^^^^^^^^^^^^ unused

",
        );
    }

    #[test]
    fn unexpected_template() {
        assert_parse_err(
            "
fn vertex() -> vec4<f32> {
    return vec4<f32>;
}
",
            "\
error: unexpected template
  ┌─ wgsl:3:12
  │
3 │     return vec4<f32>;
  │            ^^^^^^^^^ expected identifier

",
        );
    }

    #[test]
    fn expected_template_arg() {
        assert_parse_err(
            "
fn test() {
    bitcast(8);
}
",
            "\
error: `bitcast` needs a template argument specified: `T`, a type
  ┌─ wgsl:3:5
  │
3 │     bitcast(8);
  │     ^^^^^^^ is missing a template argument

",
        );
    }
}

// tiled-fork: begin tests (subpass_input frontend)
mod subpass_input {
    use super::parse_str;
    use crate::front::wgsl::assert_parse_err;

    #[test]
    fn parse_subpass_input_types() {
        // All four subpass-input types in their plain and multisampled forms.
        parse_str("@group(0) @binding(0) var s: subpass_input<f32>;").unwrap();
        parse_str("@group(0) @binding(0) var s: subpass_input<i32>;").unwrap();
        parse_str("@group(0) @binding(0) var s: subpass_input<u32>;").unwrap();
        parse_str("@group(0) @binding(0) var s: subpass_input_multisampled<f32>;").unwrap();
        parse_str("@group(0) @binding(0) var s: subpass_input_depth;").unwrap();
        parse_str("@group(0) @binding(0) var s: subpass_input_depth_multisampled;").unwrap();
        parse_str("@group(0) @binding(0) var s: subpass_input_stencil;").unwrap();
        parse_str("@group(0) @binding(0) var s: subpass_input_stencil_multisampled;").unwrap();
    }

    #[test]
    fn subpass_load_lowers() {
        // `subpassLoad` on a single-sampled subpass input should parse and lower.
        parse_str(
            "
@group(0) @binding(0) var s: subpass_input<f32>;
@fragment
fn fs() -> @location(0) vec4<f32> {
    return subpassLoad(s);
}
",
        )
        .unwrap();
    }

    #[test]
    fn subpass_load_multisampled_takes_sample_index() {
        // Multisampled subpass inputs require a `sample_index` argument.
        parse_str(
            "
@group(0) @binding(0) var s: subpass_input_multisampled<f32>;
@fragment
fn fs() -> @location(0) vec4<f32> {
    return subpassLoad(s, 0);
}
",
        )
        .unwrap();
    }

    #[test]
    fn texture_load_on_subpass_input_is_rejected() {
        assert_parse_err(
            "
@group(0) @binding(0) var s: subpass_input<f32>;
@fragment
fn fs() -> @location(0) vec4<f32> {
    return textureLoad(s, vec2<i32>(0, 0), 0);
}
",
            "\
error: textureLoad cannot be used with input attachments
  ┌─ wgsl:5:24
  │
5 │     return textureLoad(s, vec2<i32>(0, 0), 0);
  │                        ^ use subpassLoad(input_attachment) instead

",
        );
    }

    #[test]
    fn subpass_load_on_non_subpass_input_is_rejected() {
        assert_parse_err(
            "
@group(0) @binding(0) var t: texture_2d<f32>;
@fragment
fn fs() -> @location(0) vec4<f32> {
    return subpassLoad(t);
}
",
            "\
error: subpassLoad requires an input attachment
  ┌─ wgsl:5:24
  │
5 │     return subpassLoad(t);
  │                        ^ this value is not a `subpass_input*` type

",
        );
    }

    #[test]
    fn input_attachment_index_is_deprecated() {
        assert_parse_err(
            "
@group(0) @binding(0) @input_attachment_index(0) var s: subpass_input<f32>;
",
            "\
error: `@input_attachment_index` is no longer supported
  ┌─ wgsl:2:24
  │
2 │ @group(0) @binding(0) @input_attachment_index(0) var s: subpass_input<f32>;
  │                        ^^^^^^^^^^^^^^^^^^^^^^ use `subpass_input*` types and identify inputs by `@binding`

",
        );
    }
}
// tiled-fork: end tests (subpass_input frontend)

// tiled-fork: begin tests (color_attribute frontend)
mod color_attribute {
    use super::parse_str;
    use crate::front::wgsl::assert_parse_err;

    #[test]
    fn color_attribute_on_fragment_arg_parses_and_lowers() {
        // `@color(0)` on a fragment-stage entry-point argument is accepted and
        // lowers to `Binding::ColorAttachmentRead { attachment: 0 }`.
        let module = parse_str(
            "
@fragment
fn fs(@color(0) prev: vec4<f32>) -> @location(0) vec4<f32> {
    return prev;
}
",
        )
        .unwrap();

        let entry = module
            .entry_points
            .iter()
            .find(|ep| ep.name == "fs")
            .expect("fragment entry point");
        let arg = entry
            .function
            .arguments
            .first()
            .expect("entry-point argument");
        match arg.binding {
            Some(crate::Binding::ColorAttachmentRead { attachment, .. }) => {
                assert_eq!(attachment, 0);
            }
            ref other => panic!("expected ColorAttachmentRead, got {other:?}"),
        }
    }

    #[test]
    fn color_attribute_on_vertex_arg_is_rejected() {
        // `@color(N)` is only allowed on fragment-stage entry points; on a
        // vertex entry-point arg it is reported as an unknown attribute.
        assert_parse_err(
            "
@vertex
fn vs(@color(0) prev: vec4<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
",
            "\
error: unknown attribute: `color`
  ┌─ wgsl:3:8
  │
3 │ fn vs(@color(0) prev: vec4<f32>) -> @builtin(position) vec4<f32> {
  │        ^^^^^ unknown attribute

",
        );
    }

    #[test]
    fn color_attribute_on_struct_member_is_rejected() {
        // Struct members never accept `@color`, even when the struct is later
        // used as a fragment-stage input.
        assert_parse_err(
            "
struct FsIn {
    @color(0) prev: vec4<f32>,
}
",
            "\
error: @color is only valid on fragment-entry parameters, not struct members
  ┌─ wgsl:3:6
  │
3 │     @color(0) prev: vec4<f32>,
  │      ^^^^^ invalid @color placement

",
        );
    }

    #[test]
    fn color_attribute_on_fragment_return_is_rejected() {
        // Function results never accept `@color`; the binding parser is
        // invoked with `allow_color = false` for the return-binding path.
        assert_parse_err(
            "
@fragment
fn fs() -> @color(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
",
            "\
error: unknown attribute: `color`
  ┌─ wgsl:3:13
  │
3 │ fn fs() -> @color(0) vec4<f32> {
  │             ^^^^^ unknown attribute

",
        );
    }

    #[cfg(wgsl_out)]
    #[test]
    fn color_attribute_round_trips_through_wgsl_backend() {
        // The WGSL backend re-emits `@color(N)` on the lowered fragment arg,
        // and the re-parsed shader contains the original attribute.
        let source = "
@fragment
fn fs(@color(2) prev: vec4<f32>) -> @location(0) vec4<f32> {
    return prev;
}
";
        let module = parse_str(source).unwrap();
        let info = crate::valid::Validator::new(
            crate::valid::ValidationFlags::all(),
            crate::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("module is valid");

        let mut out = alloc::string::String::new();
        let mut writer =
            crate::back::wgsl::Writer::new(&mut out, crate::back::wgsl::WriterFlags::empty());
        writer.write(&module, &info).expect("wgsl write succeeds");
        assert!(
            out.contains("@color(2)"),
            "expected `@color(2)` in WGSL output, got:\n{out}"
        );

        // Re-parse the emitted WGSL to confirm it round-trips cleanly.
        parse_str(&out).expect("re-parse round-tripped WGSL");
    }
}
// tiled-fork: end tests (color_attribute frontend)
