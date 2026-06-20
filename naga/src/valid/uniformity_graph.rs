use alloc::{collections::VecDeque, vec, vec::Vec};

use crate::diagnostic_filter::{DiagnosticFilterNode, Severity, StandardFilterableTriggeringRule};
use crate::span::{AddSpan as _, WithSpan};
use crate::{arena::Handle, proc::ResolveContext, FastHashMap};

use super::analyzer::{FunctionInfo, UniformityDisruptor};
use super::{FunctionError, UniformityRequirements};

pub type NodeId = u32;

#[derive(Clone)]
pub struct UNode {
    pub edges: Vec<NodeId>,
    pub affects_cf: bool,
    pub cause: Option<Handle<crate::Expression>>,
}

struct Required {
    error: NodeId,
    warning: NodeId,
    info: NodeId,
}

#[derive(Clone, Debug, Default)]
pub(super) struct Summary {
    pub callsite_required: Option<Severity>,
    pub result_may_be_non_uniform: bool,
    pub parameters: Vec<ParameterSummary>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct ParameterSummary {
    pub is_pointer: bool,
    pub tag_direct: ParameterTag,
    pub tag_retval: ParameterTag,
    pub pointer_may_become_non_uniform: bool,
    pub ptr_output_source_param_values: Vec<usize>,
    pub ptr_output_source_param_contents: Vec<usize>,
}

impl ParameterSummary {
    fn is_pointer(&self) -> bool {
        self.is_pointer
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ParameterTag {
    pub kind: ParameterTagKind,
    pub severity: Option<Severity>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum ParameterTagKind {
    ValueRequiredToBeUniform,
    ContentsRequiredToBeUniform,
    #[default]
    NoRestriction,
}

impl Required {
    fn get(&self, severity: Severity) -> Option<NodeId> {
        match severity {
            Severity::Off => None,
            Severity::Info => Some(self.info),
            Severity::Warning => Some(self.warning),
            Severity::Error => Some(self.error),
        }
    }
}

#[derive(Clone, Copy)]
struct Flow {
    cf: NodeId,
    next: bool,
    escapes_function: bool,
}

#[derive(Clone, Copy)]
enum ControlKind {
    Loop,
    Switch,
}

struct ControlContext {
    kind: ControlKind,
    exit_values: FastHashMap<Handle<crate::LocalVariable>, Vec<NodeId>>,
    exit_cfs: Vec<NodeId>,
    continue_values: FastHashMap<Handle<crate::LocalVariable>, Vec<NodeId>>,
    continue_cfs: Vec<NodeId>,
    escapes_function: bool,
}

struct ParameterNodes {
    value: NodeId,
    ptr_input_contents: Option<NodeId>,
    ptr_output_contents: Option<NodeId>,
}

impl ControlContext {
    fn new(kind: ControlKind) -> Self {
        Self {
            kind,
            exit_values: FastHashMap::default(),
            exit_cfs: Vec::new(),
            continue_values: FastHashMap::default(),
            continue_cfs: Vec::new(),
            escapes_function: false,
        }
    }
}

struct Builder<'a> {
    nodes: Vec<UNode>,
    required: Required,
    barrier_required: NodeId,
    may_be_non_uniform: NodeId,
    cf_start: NodeId,
    value_return: NodeId,
    variables: FastHashMap<Handle<crate::LocalVariable>, NodeId>,
    pointer_expr_node: FastHashMap<Handle<crate::Expression>, NodeId>,
    expr_node: FastHashMap<Handle<crate::Expression>, NodeId>,
    parameters: Vec<ParameterNodes>,
    pointer_parameter_contents: Vec<Option<NodeId>>,
    control_stack: Vec<ControlContext>,
    fun: &'a crate::Function,
    module: &'a crate::Module,
    other_functions: &'a [FunctionInfo],
    resolve_context: &'a ResolveContext<'a>,
    diagnostic_filter_leaf: Option<Handle<DiagnosticFilterNode>>,
}

pub(super) fn validate_function(
    fun: &crate::Function,
    module: &crate::Module,
    other_functions: &[FunctionInfo],
    info: &FunctionInfo,
    resolve_context: &ResolveContext,
) -> Result<Summary, WithSpan<FunctionError>> {
    let mut builder = Builder::new(fun, module, other_functions, resolve_context);
    builder.initialize_parameters();
    builder.initialize_locals();
    builder.process_block(&fun.body, builder.cf_start);
    builder.finish_pointer_parameter_outputs();
    builder.emit_diagnostics(info)?;
    Ok(builder.compute_summary())
}

impl<'a> Builder<'a> {
    fn new(
        fun: &'a crate::Function,
        module: &'a crate::Module,
        other_functions: &'a [FunctionInfo],
        resolve_context: &'a ResolveContext<'a>,
    ) -> Self {
        let mut nodes = Vec::new();
        let required_error = add_node(&mut nodes, false, None);
        let required_warning = add_node(&mut nodes, false, None);
        let required_info = add_node(&mut nodes, false, None);
        let barrier_required = add_node(&mut nodes, false, None);
        let may_be_non_uniform = add_node(&mut nodes, false, None);
        let cf_start = add_node(&mut nodes, false, None);
        let value_return = add_node(&mut nodes, false, None);

        Self {
            nodes,
            required: Required {
                error: required_error,
                warning: required_warning,
                info: required_info,
            },
            barrier_required,
            may_be_non_uniform,
            cf_start,
            value_return,
            variables: FastHashMap::default(),
            pointer_expr_node: FastHashMap::default(),
            expr_node: FastHashMap::default(),
            parameters: Vec::new(),
            pointer_parameter_contents: Vec::new(),
            control_stack: Vec::new(),
            fun,
            module,
            other_functions,
            resolve_context,
            diagnostic_filter_leaf: fun.diagnostic_filter_leaf,
        }
    }

    fn initialize_parameters(&mut self) {
        for (index, argument) in self.resolve_context.arguments.iter().enumerate() {
            let value = self.fresh(None);
            let (ptr_input_contents, ptr_output_contents) = if self.type_is_pointer(argument.ty) {
                let input = self.fresh(None);
                let output = self.fresh(None);
                (Some(input), Some(output))
            } else {
                (None, None)
            };
            self.parameters.push(ParameterNodes {
                value,
                ptr_input_contents,
                ptr_output_contents,
            });
            self.pointer_parameter_contents.push(ptr_input_contents);

            if let Some(input) = ptr_input_contents {
                let _ = self.pointer_parameter_contents[index].insert(input);
            }
        }
    }

    fn initialize_locals(&mut self) {
        for (handle, local) in self.fun.local_variables.iter() {
            let node = self.fresh(None);
            if let Some(init) = local.init {
                let init_node = if self.type_is_pointer(local.ty) {
                    self.pointer_value_node(init)
                } else {
                    self.value_node(init)
                };
                self.edge(node, init_node);
            }
            self.variables.insert(handle, node);
        }
    }

    fn finish_pointer_parameter_outputs(&mut self) {
        for index in 0..self.parameters.len() {
            let Some(output) = self.parameters[index].ptr_output_contents else {
                continue;
            };
            if let Some(contents) = self.pointer_parameter_contents[index] {
                self.edge(output, contents);
            }
        }
    }

    fn process_block(&mut self, block: &crate::Block, mut cf: NodeId) -> Flow {
        let mut escapes_function = false;
        for statement in block.iter() {
            let mut flow = self.process_statement(statement, cf);
            escapes_function |= flow.escapes_function;
            cf = flow.cf;
            if !flow.next {
                flow.escapes_function = escapes_function;
                return flow;
            }
        }
        Flow {
            cf,
            next: true,
            escapes_function,
        }
    }

    fn process_statement(&mut self, statement: &crate::Statement, cf: NodeId) -> Flow {
        use crate::Statement as S;

        let next = |cf| Flow {
            cf,
            next: true,
            escapes_function: false,
        };
        match *statement {
            S::Emit(ref range) => {
                for expr in range.clone() {
                    self.compute_expr_node(expr, cf);
                }
                next(cf)
            }
            S::Block(ref block) => self.process_block(block, cf),
            S::If {
                condition,
                ref accept,
                ref reject,
            } => self.process_if(condition, accept, reject, cf),
            S::Switch {
                selector,
                ref cases,
            } => self.process_switch(selector, cases, cf),
            S::Loop {
                ref body,
                ref continuing,
                break_if,
            } => self.process_loop(body, continuing, break_if, cf),
            S::Break => {
                self.record_break(cf);
                Flow {
                    cf,
                    next: false,
                    escapes_function: false,
                }
            }
            S::Continue => {
                self.record_continue(cf);
                Flow {
                    cf,
                    next: false,
                    escapes_function: false,
                }
            }
            S::Kill => Flow {
                cf,
                next: false,
                escapes_function: true,
            },
            S::Return { value } => {
                if let Some(value) = value {
                    let value_node = self.value_node(value);
                    self.edge(self.value_return, value_node);
                }
                Flow {
                    cf,
                    next: false,
                    escapes_function: true,
                }
            }
            S::ControlBarrier(_) | S::MemoryBarrier(_) => {
                self.edge(self.barrier_required, cf);
                next(cf)
            }
            S::WorkGroupUniformLoad { pointer, result } => {
                let pointer_node = self.pointer_value_node(pointer);
                self.edge(self.barrier_required, cf);
                self.edge(self.barrier_required, pointer_node);
                let _ = self.expr_node_or_fresh(result);
                next(cf)
            }
            S::Store { pointer, value } => {
                self.process_store(pointer, value);
                next(cf)
            }
            S::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                let _ = self.value_node(image);
                let _ = self.value_node(coordinate);
                if let Some(array_index) = array_index {
                    let _ = self.value_node(array_index);
                }
                let _ = self.value_node(value);
                next(cf)
            }
            S::Call {
                function,
                ref arguments,
                result,
            } => {
                self.process_call(function, arguments, result, cf);
                next(cf)
            }
            S::Atomic {
                pointer,
                ref fun,
                value,
                result,
            } => {
                let _ = self.value_node(pointer);
                let _ = self.value_node(value);
                if let crate::AtomicFunction::Exchange { compare: Some(cmp) } = *fun {
                    let _ = self.value_node(cmp);
                }
                if let Some(result) = result {
                    let result_node = self.expr_node_or_fresh(result);
                    self.edge(result_node, self.may_be_non_uniform);
                }
                next(cf)
            }
            S::ImageAtomic {
                image,
                coordinate,
                array_index,
                fun: _,
                value,
            } => {
                let _ = self.value_node(image);
                let _ = self.value_node(coordinate);
                if let Some(array_index) = array_index {
                    let _ = self.value_node(array_index);
                }
                let _ = self.value_node(value);
                next(cf)
            }
            S::RayQuery { query, ref fun } => {
                let _ = self.value_node(query);
                match *fun {
                    crate::RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    } => {
                        let _ = self.value_node(acceleration_structure);
                        let _ = self.value_node(descriptor);
                    }
                    crate::RayQueryFunction::Proceed { result } => {
                        let result_node = self.expr_node_or_fresh(result);
                        self.edge(result_node, self.may_be_non_uniform);
                    }
                    crate::RayQueryFunction::GenerateIntersection { hit_t } => {
                        let _ = self.value_node(hit_t);
                    }
                    crate::RayQueryFunction::ConfirmIntersection
                    | crate::RayQueryFunction::Terminate => {}
                }
                next(cf)
            }
            S::RayPipelineFunction(ref fun) => {
                match *fun {
                    crate::RayPipelineFunction::TraceRay {
                        acceleration_structure,
                        descriptor,
                        payload,
                    } => {
                        let _ = self.value_node(acceleration_structure);
                        let _ = self.value_node(descriptor);
                        let _ = self.value_node(payload);
                    }
                }
                next(cf)
            }
            S::SubgroupBallot { result, predicate } => {
                if let Some(predicate) = predicate {
                    let _ = self.value_node(predicate);
                }
                let result_node = self.expr_node_or_fresh(result);
                self.edge(result_node, self.may_be_non_uniform);
                next(cf)
            }
            S::SubgroupGather {
                ref mode,
                argument,
                result,
            } => {
                let _ = self.value_node(argument);
                self.process_gather_mode(mode);
                let result_node = self.expr_node_or_fresh(result);
                self.edge(result_node, self.may_be_non_uniform);
                next(cf)
            }
            S::SubgroupCollectiveOperation {
                op: _,
                collective_op: _,
                argument,
                result,
            } => {
                let _ = self.value_node(argument);
                let result_node = self.expr_node_or_fresh(result);
                self.edge(result_node, self.may_be_non_uniform);
                next(cf)
            }
            S::CooperativeStore { target, ref data } => {
                let _ = self.value_node(target);
                let _ = self.value_node(data.pointer);
                let _ = self.value_node(data.stride);
                next(cf)
            }
        }
    }

    fn process_if(
        &mut self,
        condition: Handle<crate::Expression>,
        accept: &crate::Block,
        reject: &crate::Block,
        cf: NodeId,
    ) -> Flow {
        let condition_node = self.value_node(condition);
        let if_cf = self.fresh_cf(Some(condition));
        self.edge(if_cf, condition_node);
        self.edge(if_cf, cf);

        let before = self.variables.clone();

        self.variables = before.clone();
        let accept_flow = self.process_block(accept, if_cf);
        let accept_vars = self.variables.clone();

        self.variables = before.clone();
        let reject_flow = self.process_block(reject, if_cf);
        let reject_vars = self.variables.clone();

        let mut merged = before.clone();
        for (&local, &pre_node) in before.iter() {
            let accept_node = accept_vars.get(&local).copied().unwrap_or(pre_node);
            let reject_node = reject_vars.get(&local).copied().unwrap_or(pre_node);
            if accept_node != pre_node || reject_node != pre_node {
                let exit = self.fresh(None);
                if accept_flow.next {
                    self.edge(exit, accept_node);
                    self.edge(exit, accept_flow.cf);
                }
                if reject_flow.next {
                    self.edge(exit, reject_node);
                    self.edge(exit, reject_flow.cf);
                }
                merged.insert(local, exit);
            }
        }
        self.variables = merged;

        let escapes_function = accept_flow.escapes_function || reject_flow.escapes_function;
        match (accept_flow.next, reject_flow.next, escapes_function) {
            (true, true, false) => Flow {
                cf,
                next: true,
                escapes_function: false,
            },
            (true, true, true) => {
                let cf_end = self.fresh_cf(None);
                self.edge(cf_end, accept_flow.cf);
                self.edge(cf_end, reject_flow.cf);
                Flow {
                    cf: cf_end,
                    next: true,
                    escapes_function,
                }
            }
            (true, false, _) => Flow {
                cf: {
                    let cf_end = self.fresh_cf(None);
                    self.edge(cf_end, accept_flow.cf);
                    self.edge(cf_end, reject_flow.cf);
                    cf_end
                },
                next: true,
                escapes_function,
            },
            (false, true, _) => Flow {
                cf: {
                    let cf_end = self.fresh_cf(None);
                    self.edge(cf_end, accept_flow.cf);
                    self.edge(cf_end, reject_flow.cf);
                    cf_end
                },
                next: true,
                escapes_function,
            },
            (false, false, _) => {
                let cf_end = self.fresh_cf(None);
                self.edge(cf_end, accept_flow.cf);
                self.edge(cf_end, reject_flow.cf);
                Flow {
                    cf: cf_end,
                    next: false,
                    escapes_function,
                }
            }
        }
    }

    fn process_switch(
        &mut self,
        selector: Handle<crate::Expression>,
        cases: &[crate::SwitchCase],
        cf: NodeId,
    ) -> Flow {
        let selector_node = self.value_node(selector);
        let switch_cf = self.fresh_cf(Some(selector));
        self.edge(switch_cf, selector_node);
        self.edge(switch_cf, cf);

        let before = self.variables.clone();
        self.control_stack
            .push(ControlContext::new(ControlKind::Switch));

        let mut fallthrough_vars: Option<FastHashMap<Handle<crate::LocalVariable>, NodeId>> = None;
        let mut fallthrough_cf = switch_cf;

        for case in cases {
            self.variables = fallthrough_vars.take().unwrap_or_else(|| before.clone());
            let case_cf = if fallthrough_cf == switch_cf {
                switch_cf
            } else {
                fallthrough_cf
            };
            let flow = self.process_block(&case.body, case_cf);
            if let Some(info) = self.control_stack.last_mut() {
                info.escapes_function |= flow.escapes_function;
            }

            if flow.next {
                if case.fall_through {
                    fallthrough_vars = Some(self.variables.clone());
                    fallthrough_cf = flow.cf;
                } else {
                    self.record_switch_exit(flow.cf);
                    fallthrough_cf = switch_cf;
                }
            } else {
                fallthrough_cf = switch_cf;
            }
        }

        let info = self.control_stack.pop().unwrap();
        self.variables = before;
        let switch_escapes_function = info.escapes_function;
        if self.install_exit_values(info) {
            if switch_escapes_function {
                let cf_end = self.fresh_cf(None);
                self.edge(cf_end, switch_cf);
                Flow {
                    cf: cf_end,
                    next: true,
                    escapes_function: true,
                }
            } else {
                Flow {
                    cf,
                    next: true,
                    escapes_function: false,
                }
            }
        } else {
            Flow {
                cf: self.fresh_cf(None),
                next: false,
                escapes_function: switch_escapes_function,
            }
        }
    }

    fn process_loop(
        &mut self,
        body: &crate::Block,
        continuing: &crate::Block,
        break_if: Option<Handle<crate::Expression>>,
        cf: NodeId,
    ) -> Flow {
        let before = self.variables.clone();
        let loop_start = self.fresh_cf(None);
        self.edge(loop_start, cf);

        let mut in_nodes = FastHashMap::default();
        for (&local, &value) in before.iter() {
            let in_node = self.fresh(None);
            self.edge(in_node, value);
            in_nodes.insert(local, in_node);
            self.variables.insert(local, in_node);
        }

        self.control_stack
            .push(ControlContext::new(ControlKind::Loop));
        let body_flow = self.process_block(body, loop_start);
        let body_vars = self.variables.clone();
        let body_can_reach_latch = body_flow.next;

        let mut loop_escapes_function = body_flow.escapes_function;
        let mut latch_reachable = body_can_reach_latch;
        let mut latch_cf = body_flow.cf;

        let has_continue = self
            .control_stack
            .last()
            .is_some_and(|info| !info.continue_cfs.is_empty());

        if body_can_reach_latch || has_continue {
            let continue_values = self
                .control_stack
                .last()
                .map(|info| info.continue_values.clone())
                .unwrap_or_default();
            let continue_cfs = self
                .control_stack
                .last()
                .map(|info| info.continue_cfs.clone())
                .unwrap_or_default();

            self.variables = if body_can_reach_latch {
                body_vars.clone()
            } else {
                before
                    .iter()
                    .map(|(&local, &value)| (local, value))
                    .collect()
            };

            if has_continue {
                let cf_continue = self.fresh_cf(None);
                for cf_continue_edge in continue_cfs {
                    self.edge(cf_continue, cf_continue_edge);
                }
                if body_can_reach_latch {
                    self.edge(cf_continue, body_flow.cf);
                }
                latch_cf = cf_continue;

                for (&local, &in_node) in in_nodes.iter() {
                    let continue_in = self.fresh(None);
                    if body_can_reach_latch {
                        if let Some(&body_value) = body_vars.get(&local) {
                            self.edge(continue_in, body_value);
                        }
                    }
                    if let Some(values) = continue_values.get(&local) {
                        for &value in values {
                            self.edge(continue_in, value);
                        }
                    }
                    if self.nodes[continue_in as usize].edges.is_empty() {
                        self.edge(continue_in, in_node);
                    }
                    self.variables.insert(local, continue_in);
                }
            }

            if !continuing.is_empty() {
                let continuing_flow = self.process_block(continuing, latch_cf);
                loop_escapes_function |= continuing_flow.escapes_function;
                latch_reachable = continuing_flow.next;
                latch_cf = continuing_flow.cf;
            }

            if let Some(condition) = break_if {
                let condition_node = self.value_node(condition);
                let break_if_cf = self.fresh_cf(Some(condition));
                self.edge(break_if_cf, condition_node);
                self.edge(break_if_cf, latch_cf);
                self.record_loop_exit(break_if_cf);
                latch_cf = break_if_cf;
                latch_reachable = true;
            }

            if latch_reachable {
                self.edge(loop_start, latch_cf);
                for (&local, &in_node) in in_nodes.iter() {
                    if let Some(&out_node) = self.variables.get(&local) {
                        if out_node != in_node {
                            self.edge(in_node, out_node);
                        }
                    }
                }
            }
        }

        let info = self.control_stack.pop().unwrap();
        self.variables = before;
        if self.install_exit_values(info) {
            if loop_escapes_function {
                Flow {
                    cf: latch_cf,
                    next: true,
                    escapes_function: true,
                }
            } else {
                Flow {
                    cf,
                    next: true,
                    escapes_function: false,
                }
            }
        } else {
            Flow {
                cf: latch_cf,
                next: false,
                escapes_function: loop_escapes_function,
            }
        }
    }

    fn record_break(&mut self, cf: NodeId) {
        if let Some(index) = self
            .control_stack
            .iter()
            .rposition(|info| matches!(info.kind, ControlKind::Loop | ControlKind::Switch))
        {
            let values = self.current_variable_values();
            let info = &mut self.control_stack[index];
            info.exit_cfs.push(cf);
            for (local, value) in values {
                info.exit_values.entry(local).or_default().push(value);
            }
        }
    }

    fn record_continue(&mut self, cf: NodeId) {
        if let Some(index) = self
            .control_stack
            .iter()
            .rposition(|info| matches!(info.kind, ControlKind::Loop))
        {
            let values = self.current_variable_values();
            let info = &mut self.control_stack[index];
            info.continue_cfs.push(cf);
            for (local, value) in values {
                info.continue_values.entry(local).or_default().push(value);
            }
        }
    }

    fn record_switch_exit(&mut self, cf: NodeId) {
        if let Some(index) = self
            .control_stack
            .iter()
            .rposition(|info| matches!(info.kind, ControlKind::Switch))
        {
            let values = self.current_variable_values();
            let info = &mut self.control_stack[index];
            info.exit_cfs.push(cf);
            for (local, value) in values {
                info.exit_values.entry(local).or_default().push(value);
            }
        }
    }

    fn record_loop_exit(&mut self, cf: NodeId) {
        if let Some(index) = self
            .control_stack
            .iter()
            .rposition(|info| matches!(info.kind, ControlKind::Loop))
        {
            let values = self.current_variable_values();
            let info = &mut self.control_stack[index];
            info.exit_cfs.push(cf);
            for (local, value) in values {
                info.exit_values.entry(local).or_default().push(value);
            }
        }
    }

    fn current_variable_values(&self) -> Vec<(Handle<crate::LocalVariable>, NodeId)> {
        self.variables
            .iter()
            .map(|(&local, &value)| (local, value))
            .collect()
    }

    fn install_exit_values(&mut self, info: ControlContext) -> bool {
        if info.exit_cfs.is_empty() {
            return false;
        }

        for (local, values) in info.exit_values {
            let exit = self.fresh(None);
            for value in values {
                self.edge(exit, value);
            }
            for &cf in &info.exit_cfs {
                self.edge(exit, cf);
            }
            self.variables.insert(local, exit);
        }

        true
    }

    fn process_store(
        &mut self,
        pointer: Handle<crate::Expression>,
        value: Handle<crate::Expression>,
    ) {
        let value_node = self.value_node(value);
        let pointer_node = self.pointer_value_node(pointer);
        let Some((root, partial)) = self.pointee_root(pointer) else {
            return;
        };

        match root {
            PointerRoot::Local(local) => {
                let new_node = self.fresh(None);
                self.edge(new_node, value_node);
                self.edge(new_node, pointer_node);
                if partial {
                    if let Some(previous) = self.variables.get(&local).copied() {
                        self.edge(new_node, previous);
                    }
                }
                self.variables.insert(local, new_node);
            }
            PointerRoot::Global(_) => {}
            PointerRoot::Argument(index) => {
                let new_node = self.fresh(None);
                self.edge(new_node, value_node);
                self.edge(new_node, pointer_node);
                if partial {
                    if let Some(previous) = self
                        .pointer_parameter_contents
                        .get(index)
                        .and_then(|contents| *contents)
                    {
                        self.edge(new_node, previous);
                    }
                }
                if let Some(contents) = self.pointer_parameter_contents.get_mut(index) {
                    *contents = Some(new_node);
                }
            }
        }
    }

    fn process_call(
        &mut self,
        function: Handle<crate::Function>,
        arguments: &[Handle<crate::Expression>],
        result: Option<Handle<crate::Expression>>,
        cf: NodeId,
    ) {
        let Some(summary) = self.other_functions[function.index()].graph_summary() else {
            return;
        };

        let call_node = self.fresh_cf(None);
        self.edge(call_node, cf);

        let mut arg_values = Vec::with_capacity(arguments.len());
        let mut arg_contents = Vec::with_capacity(arguments.len());
        for (index, &argument) in arguments.iter().enumerate() {
            let is_pointer = summary
                .parameters
                .get(index)
                .is_some_and(|param| param.is_pointer());
            let arg_node = self.fresh(Some(argument));
            let value = if is_pointer {
                self.pointer_value_node(argument)
            } else {
                self.value_node(argument)
            };
            self.edge(arg_node, value);
            arg_values.push(arg_node);

            if is_pointer {
                let contents = self.pointer_contents_node(argument);
                let node = self.fresh(Some(argument));
                self.edge(node, contents);
                self.edge(node, arg_node);
                arg_contents.push(Some(node));
            } else {
                arg_contents.push(None);
            }
        }

        let result_node = result.map(|result| self.expr_node_or_fresh(result));
        if summary.result_may_be_non_uniform {
            if let Some(result_node) = result_node {
                self.edge(result_node, self.may_be_non_uniform);
            }
        }

        for (index, param) in summary.parameters.iter().enumerate() {
            let Some(&arg_value) = arg_values.get(index) else {
                break;
            };

            self.apply_parameter_direct_tag(param.tag_direct, arg_value, arg_contents[index]);

            if let Some(result_node) = result_node {
                match param.tag_retval.kind {
                    ParameterTagKind::ValueRequiredToBeUniform => self.edge(result_node, arg_value),
                    ParameterTagKind::ContentsRequiredToBeUniform => {
                        if let Some(contents) = arg_contents[index] {
                            self.edge(result_node, contents);
                        }
                    }
                    ParameterTagKind::NoRestriction => {}
                }
            }

            if param.is_pointer() {
                let ptr_result = self.fresh(arguments.get(index).copied());
                if param.pointer_may_become_non_uniform {
                    self.edge(ptr_result, self.may_be_non_uniform);
                } else {
                    self.edge(ptr_result, call_node);
                    for &source in &param.ptr_output_source_param_values {
                        if let Some(&source_value) = arg_values.get(source) {
                            self.edge(ptr_result, source_value);
                        }
                    }
                    for &source in &param.ptr_output_source_param_contents {
                        if let Some(Some(source_contents)) = arg_contents.get(source).copied() {
                            self.edge(ptr_result, source_contents);
                        }
                    }
                }

                if let Some(&argument) = arguments.get(index) {
                    if let Some((root, partial)) = self.pointee_root(argument) {
                        if partial {
                            let old_value = self.contents_for_root(root);
                            self.edge(ptr_result, old_value);
                        }
                        self.update_root_contents(root, ptr_result);
                    }
                }
            }
        }

        if let Some(severity) = summary.callsite_required {
            if let Some(required) = self.required.get(severity) {
                self.edge(required, call_node);
            }
        }
    }

    fn apply_parameter_direct_tag(
        &mut self,
        tag: ParameterTag,
        value: NodeId,
        contents: Option<NodeId>,
    ) {
        let Some(severity) = tag.severity else {
            return;
        };
        let Some(required) = self.required.get(severity) else {
            return;
        };
        match tag.kind {
            ParameterTagKind::ValueRequiredToBeUniform => self.edge(required, value),
            ParameterTagKind::ContentsRequiredToBeUniform => {
                if let Some(contents) = contents {
                    self.edge(required, contents);
                }
            }
            ParameterTagKind::NoRestriction => {}
        }
    }

    fn process_gather_mode(&mut self, mode: &crate::GatherMode) {
        match *mode {
            crate::GatherMode::BroadcastFirst | crate::GatherMode::QuadSwap(_) => {}
            crate::GatherMode::Broadcast(index)
            | crate::GatherMode::Shuffle(index)
            | crate::GatherMode::ShuffleDown(index)
            | crate::GatherMode::ShuffleUp(index)
            | crate::GatherMode::ShuffleXor(index)
            | crate::GatherMode::QuadBroadcast(index) => {
                let _ = self.value_node(index);
            }
        }
    }

    fn value_node(&mut self, handle: Handle<crate::Expression>) -> NodeId {
        self.expr_node
            .get(&handle)
            .copied()
            .unwrap_or_else(|| self.compute_expr_node(handle, self.cf_start))
    }

    fn expr_node_or_fresh(&mut self, handle: Handle<crate::Expression>) -> NodeId {
        if let Some(node) = self.expr_node.get(&handle).copied() {
            node
        } else {
            let node = self.fresh(Some(handle));
            self.expr_node.insert(handle, node);
            node
        }
    }

    fn compute_expr_node(&mut self, handle: Handle<crate::Expression>, cf: NodeId) -> NodeId {
        if let Some(node) = self.expr_node.get(&handle).copied() {
            return node;
        }

        use crate::{Expression as E, SampleLevel as Sl};

        let node = match self.fun.expressions[handle] {
            E::Literal(_) | E::Constant(_) | E::Override(_) | E::ZeroValue(_) => {
                self.fresh(Some(handle))
            }
            E::Compose { ref components, .. } => {
                let node = self.fresh(Some(handle));
                for &component in components {
                    let component_node = self.value_node(component);
                    self.edge(node, component_node);
                }
                node
            }
            E::Access { base, index } => {
                let node = self.fresh(Some(handle));
                let base_node = self.value_node(base);
                let index_node = self.value_node(index);
                self.edge(node, base_node);
                self.edge(node, index_node);
                node
            }
            E::AccessIndex { base, .. } => {
                let node = self.fresh(Some(handle));
                let base_node = self.value_node(base);
                self.edge(node, base_node);
                node
            }
            E::Splat { value, .. } => self.unary_node(handle, value),
            E::Swizzle { vector, .. } => self.unary_node(handle, vector),
            E::FunctionArgument(index) => {
                let node = self.parameters[index as usize].value;
                let uniform = self
                    .resolve_context
                    .arguments
                    .get(index as usize)
                    .is_none_or(|arg| match arg.binding {
                        Some(crate::Binding::BuiltIn(
                            crate::BuiltIn::WorkGroupId
                            | crate::BuiltIn::WorkGroupSize
                            | crate::BuiltIn::NumWorkGroups,
                        )) => true,
                        Some(_) => false,
                        None => true,
                    });
                if !uniform {
                    self.edge(node, self.may_be_non_uniform);
                }
                node
            }
            E::GlobalVariable(global) => {
                let node = self.fresh(Some(handle));
                if self.global_may_be_non_uniform(global) {
                    self.edge(node, self.may_be_non_uniform);
                }
                node
            }
            E::LocalVariable(local) => self
                .variables
                .get(&local)
                .copied()
                .unwrap_or_else(|| self.fresh(Some(handle))),
            E::Load { pointer } => {
                let node = self.fresh(Some(handle));
                let contents = self.pointer_contents_node(pointer);
                let pointer_value = self.pointer_value_node(pointer);
                self.edge(node, contents);
                self.edge(node, pointer_value);
                node
            }
            E::ImageSample {
                image,
                sampler,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
                ..
            } => {
                let node = self.fresh(Some(handle));
                self.edge_operands(node, &[image, sampler, coordinate]);
                if let Some(expr) = array_index {
                    self.edge_value(node, expr);
                }
                if let Some(expr) = offset {
                    self.edge_value(node, expr);
                }
                match level {
                    Sl::Auto | Sl::Zero => {}
                    Sl::Exact(expr) | Sl::Bias(expr) => self.edge_value(node, expr),
                    Sl::Gradient { x, y } => self.edge_operands(node, &[x, y]),
                }
                if let Some(expr) = depth_ref {
                    self.edge_value(node, expr);
                }
                if level.implicit_derivatives() {
                    self.add_derivative_requirement(cf);
                }
                node
            }
            E::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => {
                let node = self.fresh(Some(handle));
                self.edge_operands(node, &[image, coordinate]);
                for expr in [array_index, sample, level].into_iter().flatten() {
                    self.edge_value(node, expr);
                }
                if self.storage_texture_load_may_be_non_uniform(image) {
                    self.edge(node, self.may_be_non_uniform);
                }
                node
            }
            E::ImageQuery { image, query } => {
                let node = self.fresh(Some(handle));
                self.edge_value(node, image);
                if let crate::ImageQuery::Size { level: Some(level) } = query {
                    self.edge_value(node, level);
                }
                node
            }
            E::Unary { expr, .. } => self.unary_node(handle, expr),
            E::Binary { left, right, .. } => {
                let node = self.fresh(Some(handle));
                self.edge_operands(node, &[left, right]);
                node
            }
            E::Select {
                condition,
                accept,
                reject,
            } => {
                let node = self.fresh(Some(handle));
                self.edge_operands(node, &[condition, accept, reject]);
                node
            }
            E::Derivative { expr, .. } => {
                let node = self.unary_node(handle, expr);
                self.add_derivative_requirement(cf);
                node
            }
            E::Relational { argument, .. } => self.unary_node(handle, argument),
            E::Math {
                arg,
                arg1,
                arg2,
                arg3,
                ..
            } => {
                let node = self.fresh(Some(handle));
                self.edge_value(node, arg);
                for expr in [arg1, arg2, arg3].into_iter().flatten() {
                    self.edge_value(node, expr);
                }
                node
            }
            E::As { expr, .. } => self.unary_node(handle, expr),
            E::CallResult(function) => {
                let node = self.fresh(Some(handle));
                if self.other_functions[function.index()]
                    .graph_summary()
                    .is_some_and(|summary| summary.result_may_be_non_uniform)
                {
                    self.edge(node, self.may_be_non_uniform);
                }
                node
            }
            E::AtomicResult { .. }
            | E::RayQueryProceedResult
            | E::SubgroupBallotResult
            | E::SubgroupOperationResult { .. } => {
                let node = self.fresh(Some(handle));
                self.edge(node, self.may_be_non_uniform);
                node
            }
            E::WorkGroupUniformLoadResult { .. } => self.fresh(Some(handle)),
            E::ArrayLength(expr) => self.unary_node(handle, expr),
            E::RayQueryGetIntersection { query, .. } => self.unary_node(handle, query),
            E::RayQueryVertexPositions { query, .. } => self.unary_node(handle, query),
            E::CooperativeLoad { ref data, .. } => {
                let node = self.fresh(Some(handle));
                self.edge_value(node, data.pointer);
                self.edge_value(node, data.stride);
                node
            }
            E::CooperativeMultiplyAdd { a, b, c } => {
                let node = self.fresh(Some(handle));
                self.edge_operands(node, &[a, b, c]);
                node
            }
            E::SubpassLoad {
                image,
                sample_index,
            } => {
                let node = self.fresh(Some(handle));
                self.edge_value(node, image);
                if let Some(sample_index) = sample_index {
                    self.edge_value(node, sample_index);
                }
                node
            }
        };

        self.expr_node.insert(handle, node);
        node
    }

    fn add_derivative_requirement(&mut self, cf: NodeId) {
        let severity = DiagnosticFilterNode::search(
            self.diagnostic_filter_leaf,
            &self.module.diagnostic_filters,
            StandardFilterableTriggeringRule::DerivativeUniformity,
        );
        if let Some(required) = self.required.get(severity) {
            self.edge(required, cf);
        }
    }

    fn unary_node(
        &mut self,
        handle: Handle<crate::Expression>,
        expr: Handle<crate::Expression>,
    ) -> NodeId {
        let node = self.fresh(Some(handle));
        let expr_node = self.value_node(expr);
        self.edge(node, expr_node);
        node
    }

    fn edge_operands(&mut self, node: NodeId, operands: &[Handle<crate::Expression>]) {
        for &operand in operands {
            self.edge_value(node, operand);
        }
    }

    fn edge_value(&mut self, node: NodeId, expr: Handle<crate::Expression>) {
        let expr_node = self.value_node(expr);
        self.edge(node, expr_node);
    }

    fn pointer_value_node(&mut self, handle: Handle<crate::Expression>) -> NodeId {
        if let Some(node) = self.pointer_expr_node.get(&handle).copied() {
            return node;
        }

        let node = match self.fun.expressions[handle] {
            crate::Expression::Access { base, index } => {
                let node = self.fresh(Some(handle));
                let base_node = self.pointer_value_node(base);
                let index_node = self.value_node(index);
                self.edge(node, base_node);
                self.edge(node, index_node);
                node
            }
            crate::Expression::AccessIndex { base, .. } => {
                let node = self.fresh(Some(handle));
                let base_node = self.pointer_value_node(base);
                self.edge(node, base_node);
                node
            }
            crate::Expression::FunctionArgument(index)
                if self
                    .resolve_context
                    .arguments
                    .get(index as usize)
                    .is_some_and(|arg| self.type_is_pointer(arg.ty)) =>
            {
                self.parameters[index as usize].value
            }
            crate::Expression::Load { .. } => self.value_node(handle),
            crate::Expression::LocalVariable(_) | crate::Expression::GlobalVariable(_) => {
                self.fresh(Some(handle))
            }
            _ => self.value_node(handle),
        };

        self.pointer_expr_node.insert(handle, node);
        node
    }

    fn pointer_contents_node(&mut self, pointer: Handle<crate::Expression>) -> NodeId {
        let node = self.fresh(Some(pointer));
        let pointer_node = self.pointer_value_node(pointer);
        self.edge(node, pointer_node);

        if let Some((root, _)) = self.pointee_root(pointer) {
            let contents = self.contents_for_root(root);
            self.edge(node, contents);
        }

        node
    }

    fn contents_for_root(&mut self, root: PointerRoot) -> NodeId {
        match root {
            PointerRoot::Local(local) => self
                .variables
                .get(&local)
                .copied()
                .unwrap_or_else(|| self.fresh(None)),
            PointerRoot::Global(global) => {
                let node = self.fresh(None);
                if self.global_may_be_non_uniform(global) {
                    self.edge(node, self.may_be_non_uniform);
                }
                node
            }
            PointerRoot::Argument(index) => self
                .pointer_parameter_contents
                .get(index)
                .and_then(|contents| *contents)
                .unwrap_or_else(|| self.fresh(None)),
        }
    }

    fn update_root_contents(&mut self, root: PointerRoot, node: NodeId) {
        match root {
            PointerRoot::Local(local) => {
                self.variables.insert(local, node);
            }
            PointerRoot::Global(_) => {}
            PointerRoot::Argument(index) => {
                if let Some(contents) = self.pointer_parameter_contents.get_mut(index) {
                    *contents = Some(node);
                }
            }
        }
    }

    fn pointee_root(&self, pointer: Handle<crate::Expression>) -> Option<(PointerRoot, bool)> {
        let mut current = pointer;
        let mut partial = false;
        loop {
            match self.fun.expressions[current] {
                crate::Expression::Access { base, .. }
                | crate::Expression::AccessIndex { base, .. } => {
                    current = base;
                    partial = true;
                }
                crate::Expression::LocalVariable(local) => {
                    return Some((PointerRoot::Local(local), partial));
                }
                crate::Expression::GlobalVariable(global) => {
                    return Some((PointerRoot::Global(global), partial));
                }
                crate::Expression::FunctionArgument(index)
                    if self
                        .resolve_context
                        .arguments
                        .get(index as usize)
                        .is_some_and(|arg| self.type_is_pointer(arg.ty)) =>
                {
                    return Some((PointerRoot::Argument(index as usize), partial));
                }
                crate::Expression::Load { pointer } => match self.pointee_root(pointer) {
                    Some((PointerRoot::Local(local), loaded_partial)) => {
                        if let Some(init) = self.fun.local_variables[local].init {
                            current = init;
                            partial |= loaded_partial;
                        } else {
                            return None;
                        }
                    }
                    Some((PointerRoot::Argument(index), loaded_partial)) => {
                        return Some((PointerRoot::Argument(index), partial || loaded_partial));
                    }
                    Some((PointerRoot::Global(_), _)) | None => return None,
                },
                _ => return None,
            }
        }
    }

    fn type_is_pointer(&self, ty: Handle<crate::Type>) -> bool {
        matches!(
            self.module.types[ty].inner,
            crate::TypeInner::Pointer { .. }
        )
    }

    fn global_may_be_non_uniform(&self, global: Handle<crate::GlobalVariable>) -> bool {
        use crate::AddressSpace as As;

        let var = &self.resolve_context.global_vars[global];
        match var.space {
            As::Function | As::Private | As::RayPayload | As::IncomingRayPayload => true,
            As::WorkGroup | As::TaskPayload => false,
            As::Uniform | As::Immediate => false,
            As::Storage { access } => access.contains(crate::StorageAccess::STORE),
            As::Handle => false,
        }
    }

    fn storage_texture_load_may_be_non_uniform(&self, image: Handle<crate::Expression>) -> bool {
        let Some(global) = self.global_root(image) else {
            return false;
        };
        let var = &self.module.global_variables[global];
        let ty = &self.module.types[var.ty].inner;
        matches!(
            ty,
            crate::TypeInner::Image {
                class: crate::ImageClass::Storage { access, .. },
                ..
            } if access.contains(crate::StorageAccess::STORE)
        )
    }

    fn global_root(
        &self,
        expr: Handle<crate::Expression>,
    ) -> Option<Handle<crate::GlobalVariable>> {
        let mut current = expr;
        loop {
            match self.fun.expressions[current] {
                crate::Expression::Access { base, .. }
                | crate::Expression::AccessIndex { base, .. } => current = base,
                crate::Expression::GlobalVariable(global) => return Some(global),
                _ => return None,
            }
        }
    }

    fn emit_diagnostics(&mut self, _info: &FunctionInfo) -> Result<(), WithSpan<FunctionError>> {
        self.emit_reachable_error(
            self.barrier_required,
            UniformityRequirements::WORK_GROUP_BARRIER,
            Severity::Error,
        )?;

        for (required, severity) in [
            (self.required.error, Severity::Error),
            (self.required.warning, Severity::Warning),
            (self.required.info, Severity::Info),
        ] {
            self.emit_reachable_error(required, UniformityRequirements::DERIVATIVE, severity)?;
        }

        Ok(())
    }

    fn compute_summary(&self) -> Summary {
        let mut summary = Summary {
            callsite_required: None,
            result_may_be_non_uniform: false,
            parameters: self
                .parameters
                .iter()
                .map(|parameter| ParameterSummary {
                    is_pointer: parameter.ptr_input_contents.is_some(),
                    ..ParameterSummary::default()
                })
                .collect(),
        };

        for (required, severity) in [
            (self.barrier_required, Severity::Error),
            (self.required.error, Severity::Error),
            (self.required.warning, Severity::Warning),
            (self.required.info, Severity::Info),
        ] {
            let reachable = self.reachable_from(required);
            if reachable[self.cf_start as usize] && summary.callsite_required.is_none() {
                summary.callsite_required = Some(severity);
            }

            for index in 0..self.parameters.len() {
                if summary.parameters[index].tag_direct.kind == ParameterTagKind::NoRestriction {
                    summary.parameters[index].tag_direct =
                        self.parameter_tag_from_reachable(&reachable, index, Some(severity));
                }
            }
        }

        let return_reachable = self.reachable_from(self.value_return);
        summary.result_may_be_non_uniform = return_reachable[self.may_be_non_uniform as usize];
        for index in 0..self.parameters.len() {
            summary.parameters[index].tag_retval =
                self.parameter_tag_from_reachable(&return_reachable, index, None);
        }

        for index in 0..self.parameters.len() {
            let Some(output) = self.parameters[index].ptr_output_contents else {
                continue;
            };
            let reachable = self.reachable_from(output);
            summary.parameters[index].pointer_may_become_non_uniform =
                reachable[self.may_be_non_uniform as usize];

            for source in 0..self.parameters.len() {
                match self
                    .parameter_tag_from_reachable(&reachable, source, None)
                    .kind
                {
                    ParameterTagKind::ValueRequiredToBeUniform => summary.parameters[index]
                        .ptr_output_source_param_values
                        .push(source),
                    ParameterTagKind::ContentsRequiredToBeUniform => summary.parameters[index]
                        .ptr_output_source_param_contents
                        .push(source),
                    ParameterTagKind::NoRestriction => {}
                }
            }
        }

        summary
    }

    fn parameter_tag_from_reachable(
        &self,
        reachable: &[bool],
        index: usize,
        severity: Option<Severity>,
    ) -> ParameterTag {
        let Some(parameter) = self.parameters.get(index) else {
            return ParameterTag::default();
        };

        if let Some(input) = parameter.ptr_input_contents {
            if reachable[input as usize] {
                return ParameterTag {
                    kind: ParameterTagKind::ContentsRequiredToBeUniform,
                    severity,
                };
            }
            if reachable[parameter.value as usize] {
                return ParameterTag {
                    kind: ParameterTagKind::ValueRequiredToBeUniform,
                    severity,
                };
            }
        } else if reachable[parameter.value as usize] {
            return ParameterTag {
                kind: ParameterTagKind::ValueRequiredToBeUniform,
                severity,
            };
        }

        ParameterTag::default()
    }

    fn reachable_from(&self, start: NodeId) -> Vec<bool> {
        let mut reachable = vec![false; self.nodes.len()];
        let mut queue = VecDeque::new();
        reachable[start as usize] = true;
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            for &edge in &self.nodes[node as usize].edges {
                if !reachable[edge as usize] {
                    reachable[edge as usize] = true;
                    queue.push_back(edge);
                }
            }
        }

        reachable
    }

    fn emit_reachable_error(
        &mut self,
        start: NodeId,
        requirements: UniformityRequirements,
        severity: Severity,
    ) -> Result<(), WithSpan<FunctionError>> {
        if let Some(cause) = self.find_cause_if_reaches_non_uniform(start) {
            severity.report_diag(
                FunctionError::NonUniformControlFlow(
                    requirements,
                    cause,
                    UniformityDisruptor::Expression(cause),
                )
                .with_span_handle(cause, &self.fun.expressions),
                |e, level| log::log!(level, "{e}"),
            )?;
        }
        Ok(())
    }

    fn find_cause_if_reaches_non_uniform(
        &mut self,
        start: NodeId,
    ) -> Option<Handle<crate::Expression>> {
        let mut visited_from = vec![None; self.nodes.len()];
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited_from[start as usize] = Some(start);

        while let Some(node) = queue.pop_front() {
            if node == self.may_be_non_uniform {
                let mut current = node;
                while current != start {
                    let graph_node = &self.nodes[current as usize];
                    if graph_node.affects_cf {
                        if let Some(cause) = graph_node.cause {
                            return Some(cause);
                        }
                    }
                    if let Some(cause) = graph_node.cause {
                        return Some(cause);
                    }
                    current = visited_from[current as usize]?;
                }
                return None;
            }

            for &edge in &self.nodes[node as usize].edges {
                if visited_from[edge as usize].is_none() {
                    visited_from[edge as usize] = Some(node);
                    queue.push_back(edge);
                }
            }
        }

        None
    }

    fn fresh(&mut self, cause: Option<Handle<crate::Expression>>) -> NodeId {
        add_node(&mut self.nodes, false, cause)
    }

    fn fresh_cf(&mut self, cause: Option<Handle<crate::Expression>>) -> NodeId {
        add_node(&mut self.nodes, true, cause)
    }

    fn edge(&mut self, from: NodeId, to: NodeId) {
        let edges = &mut self.nodes[from as usize].edges;
        if !edges.contains(&to) {
            edges.push(to);
        }
    }
}

#[derive(Clone, Copy)]
enum PointerRoot {
    Local(Handle<crate::LocalVariable>),
    Global(Handle<crate::GlobalVariable>),
    Argument(usize),
}

fn add_node(
    nodes: &mut Vec<UNode>,
    affects_cf: bool,
    cause: Option<Handle<crate::Expression>>,
) -> NodeId {
    let id = nodes.len() as NodeId;
    nodes.push(UNode {
        edges: Vec::new(),
        affects_cf,
        cause,
    });
    id
}
