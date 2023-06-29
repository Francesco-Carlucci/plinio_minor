from typing import Any, Tuple, cast
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from .nn import SuperNetCombiner
from plinio.graph.utils import NamedLeafModules, fx_to_nx_graph
from plinio.graph.inspection import is_layer, get_graph_inputs, named_leaf_modules, \
        uniquify_leaf_modules
from plinio.graph.annotation import clean_up_propagated_shapes


class SuperNetTracer(fx.Tracer):
    def __init__(self) -> None:
        super().__init__()  # type: ignore

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, SuperNetCombiner):
            return True
        # if isinstance(m, PITModule):
            # return True
        else:
            return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)


def convert(model: nn.Module, input_example: Any, conversion_type: str,
            ) -> Tuple[nn.Module, NamedLeafModules, NamedLeafModules]:
    """Converts a nn.Module, to/from "NAS-able" PIT and SuperNet format

    :param model: the input nn.Module
    :type model: nn.Module
    :param input_example: an input with the same shape and type of the seed's input, used
    for symbolic tracing (default: None)
    :type input_example: Any
    :param conversion_type: a string specifying the type of conversion. Supported types:
    ('import', 'export')
    :type conversion_type: str
    :return: the converted model, and two lists of all (or all unique) leaf modules for
    the NAS
    :rtype: Tuple[nn.Module, NamedLeafModule, NamedLeafModules]
    """

    if conversion_type not in ('import', 'export'):
        raise ValueError("Unsupported conversion type {}".format(conversion_type))

    tracer = SuperNetTracer()
    graph = tracer.trace(model.eval())
    name = model.__class__.__name__
    mod = fx.GraphModule(tracer.root, graph, name)
    if len(get_graph_inputs(mod.graph)) > 1:
        ShapeProp(mod).propagate(*input_example)
    else:
        ShapeProp(mod).propagate(input_example)
    clean_up_propagated_shapes(mod)
    if conversion_type == 'import':
        link_combiners_to_branches(mod)
    if conversion_type == 'export':
        export_graph(mod)

    mod.graph.lint()
    mod.recompile()
    nlf = named_leaf_modules(mod)
    ulf = uniquify_leaf_modules(nlf)
    return mod, nlf, ulf


def link_combiners_to_branches(mod: fx.GraphModule):
    """Associates the various SuperNet branches to their combiner"""
    # TODO: relies on the attribute names sn_combiner and sn_branches. Not nice, but didn't find
    # a better solution that remains flexible.

    # First finds all modules included in SuperNet branches
    sn_modules = {}
    g = fx_to_nx_graph(mod.graph)
    for n in g.nodes:
        path = str(n.target).split('.')
        # TODO: only considers call_module for now. Cost of call_method/call_function
        # will be ignored
        if 'sn_branches' in str(n.target) and n.op == 'call_module':
            sub_mod = mod.get_submodule(n.target)
            # skip "untyped" modules generated by fx during tracing
            if type(sub_mod) != nn.Module:
                parent_name = '.'.join(path[:path.index('sn_branches')])
                if parent_name not in sn_modules:
                    sn_modules[parent_name] = {}
                branch_id = int(path[path.index('sn_branches')+1])
                if branch_id not in sn_modules[parent_name]:
                    sn_modules[parent_name][branch_id] = []
                sn_modules[parent_name][branch_id].append((str(n.target), n, sub_mod))

    # Then passes them to the corresponding combiners
    for lname in sn_modules.keys():
        comb_lname = lname + ".sn_combiner"
        sub_mod = cast(SuperNetCombiner, mod.get_submodule(comb_lname))
        for i in range(sub_mod.n_branches):
            nlf = sn_modules[lname][i]
            ulf = uniquify_leaf_modules(nlf)
            # only pass the uniquified version, because if the whole SuperNetModule is invoked
            # multiple times in a forward pass, and the cost specification is not "shared",
            # then the combiner's get_cost method is already called multiple times.
            sub_mod.set_sn_branch(i, ulf)


def export_graph(mod: fx.GraphModule):
    """Exports the graph of the final NN, selecting the appropriate SuperNet branches.

    :param mod: a torch.fx.GraphModule of a SuperNet
    :type mod: fx.GraphModule
    """
    # TODO: relies on the attribute name sn_branches. Not nice, but didn't find
    # a better solution that remains flexible.
    for n in mod.graph.nodes:
        if is_layer(n, mod, (SuperNetCombiner,)):
            sub_mod = cast(SuperNetCombiner, mod.get_submodule(n.target))
            best_idx = sub_mod.best_layer_index()
            best_branch_name = 'sn_branches.' + str(best_idx)
            to_erase = []
            for ni in n.all_input_nodes:
                if best_branch_name in str(ni.target):
                    n.replace_all_uses_with(ni)
                else:
                    to_erase.append(ni)
            n.args = ()
            mod.graph.erase_node(n)
            for ni in to_erase:
                ni.args = ()
                mod.graph.erase_node(ni)
    mod.graph.eliminate_dead_code()
    mod.delete_all_unused_submodules()
