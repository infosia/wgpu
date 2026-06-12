struct Output {
    outer: u32,
    body: u32,
    continuing_value: u32,
    iterations: u32,
}

@group(0) @binding(0)
var<storage, read_write> output: Output;

@compute @workgroup_size(1)
fn main() {
    var my_idx = 7u;
    output.outer = my_idx;

    loop {
        var my_idx = output.iterations + 10u;
        output.body = my_idx;

        continuing {
            var my_idx = output.iterations + 20u;
            output.continuing_value = my_idx;
            output.iterations += 1u;
            break if my_idx >= 21u;
        }
    }
}
