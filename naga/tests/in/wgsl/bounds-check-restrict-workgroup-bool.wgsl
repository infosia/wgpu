var<workgroup> workgroup_bools: array<bool, 4>;

fn set_workgroup_bool_array(i: u32) {
    workgroup_bools[i] = true;
}

@compute @workgroup_size(1)
fn main() {
    set_workgroup_bool_array(10u);
}
