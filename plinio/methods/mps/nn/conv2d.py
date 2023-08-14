# *----------------------------------------------------------------------------*
# * Copyright (C) 2022 Politecnico di Torino, Italy                            *
# * SPDX-License-Identifier: Apache-2.0                                        *
# *                                                                            *
# * Licensed under the Apache License, Version 2.0 (the "License");            *
# * you may not use this file except in compliance with the License.           *
# * You may obtain a copy of the License at                                    *
# *                                                                            *
# * http://www.apache.org/licenses/LICENSE-2.0                                 *
# *                                                                            *
# * Unless required by applicable law or agreed to in writing, software        *
# * distributed under the License is distributed on an "AS IS" BASIS,          *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# * See the License for the specific language governing permissions and        *
# * limitations under the License.                                             *
# *                                                                            *
# * Author:  Matteo Risso <matteo.risso@polito.it>                             *
# *----------------------------------------------------------------------------*

from typing import Dict, Any, Iterator, Tuple, cast, Union, List
import torch
import torch.fx as fx
import torch.nn as nn
from ..quant.quantizers import Quantizer
from ..quant.nn import QuantConv2d, QuantList
from .module import MPSModule
from .qtz import MPSType, MPSPerLayerQtz, MPSPerChannelQtz, MPSBiasQtz
from plinio.graph.features_calculation import ConstFeaturesCalculator, FeaturesCalculator


class MPSConv2d(nn.Conv2d, MPSModule):
    """A nn.Module implementing a Conv2d layer with mixed-precision search support

    :param conv: the inner `nn.Conv2d` layer to be optimized
    :type conv: nn.Conv2d
    :param out_a_mps_quantizer: activation MPS quantizer
    :type out_a_mps_quantizer: MPSQtzLayer
    :param w_mps_quantizer: weight MPS quantizer
    :type w_mps_quantizer: Union[MPSQtzLayer, MPSQtzChannel]
    :param b_mps_quantizer: bias MPS quantizer
    :type b_mps_quantizer: MPSQtzBias
    """
    def __init__(self,
                 conv: nn.Conv2d,
                 out_a_mps_quantizer: MPSPerLayerQtz,
                 w_mps_quantizer: Union[MPSPerLayerQtz, MPSPerChannelQtz],
                 b_mps_quantizer: MPSBiasQtz):
        super(MPSConv2d, self).__init__(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode)
        is_depthwise = (conv.groups == conv.in_channels and conv.groups == conv.out_channels)
        if conv.groups != 1 and (not is_depthwise):
            msg = ('MPS currently supports only full or DepthWise Conv.,'
                   'not other groupwise versions')
            raise AttributeError(msg)
        with torch.no_grad():
            self.weight.copy_(conv.weight)
            if conv.bias is not None:
                self.bias = cast(torch.nn.parameter.Parameter, self.bias)
                self.bias.copy_(conv.bias)
            else:
                self.bias = None
        self.out_a_mps_quantizer = out_a_mps_quantizer
        self.w_mps_quantizer = w_mps_quantizer
        if self.bias is not None:
            self.b_mps_quantizer = b_mps_quantizer
        else:
            self.b_mps_quantizer = lambda *args: None  # Do Nothing
        # this will be overwritten later when we process the model graph
        self._input_features_calculator = ConstFeaturesCalculator(conv.in_channels)
        # this will be overwritten later when we process the model graph
        self.in_a_mps_quantizer = cast(MPSPerLayerQtz, nn.Identity())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the mixed-precision NAS-able layer.

        In a nutshell,:
        - Quantize and combine the weight tensor at the different `precisions`.
        - Quantize and combine the bias tensor at fixed precision.
        - Compute Conv2d operation using the previous obtained quantized tensors.
        - Quantize and combine the output tensor at the different `precisions`.

        :param input: the input activations tensor
        :type input: torch.Tensor
        :return: the output activations tensor
        :rtype: torch.Tensor
        """
        # Quantization of weight and bias
        q_w = self.w_mps_quantizer(self.weight)
        q_b = self.b_mps_quantizer(self.bias)
        # Linear operation
        out = self._conv_forward(input, q_w, q_b)
        # Quantization of output
        q_out = self.out_a_mps_quantizer(out)
        return q_out

    @staticmethod
    def autoimport(n: fx.Node,
                   mod: fx.GraphModule,
                   out_a_mps_quantizer: MPSPerLayerQtz,
                   w_mps_quantizer: Union[MPSPerLayerQtz, MPSPerChannelQtz],
                   b_mps_quantizer: MPSBiasQtz):
        """Create a new fx.Node relative to a MPSConv2d layer, starting from the fx.Node
        of a nn.Conv2d layer, and replace it into the parent fx.GraphModule

        :param n: a fx.Node corresponding to a nn.Conv2d layer
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param out_a_mps_quantizer: The MPS quantizer to be used for activations
        :type out_a_mps_quantizer: MPSQtzLayer
        :param w_mps_quantizer: The MPS quantizer to be used for weights
        :type w_mps_quantizer: Union[MPSQtzLayer, MPSQtzChannel]
        :param b_mps_quantizer: The MPS quantizer to be used for biases (if present)
        :type b_mps_quantizer: MPSQtzBias
        :raises TypeError: if the input fx.Node is not of the correct type
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != nn.Conv2d:
            msg = f"Trying to generate MPSConv2d from layer of type {type(submodule)}"
            raise TypeError(msg)
        submodule = cast(nn.Conv2d, submodule)
        new_submodule = MPSConv2d(submodule,
                                  out_a_mps_quantizer,
                                  w_mps_quantizer,
                                  b_mps_quantizer)
        mod.add_submodule(str(n.target), new_submodule)

    @staticmethod
    def export(n: fx.Node, mod: fx.GraphModule):
        """Replaces a fx.Node corresponding to a MPSConv2d layer,
        with the selected fake-quantized nn.Conv2d layer within a fx.GraphModule

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != MPSConv2d:
            raise TypeError(f"Trying to export a layer of type {type(submodule)}")

        # per-layer search => single precision/quantizer
        if isinstance(submodule.w_mps_quantizer, MPSPerLayerQtz):
            if submodule.bias is not None:
                # TODO: DP not sure why bias quantizer was re-created here,
                # trying to use the existing one now...
                b_quantizer = cast(Quantizer, submodule.b_mps_quantizer.qtz_func)
            else:
                b_quantizer = None
            new_submodule = QuantConv2d(submodule,
                                        submodule.selected_in_a_quantizer,
                                        submodule.selected_out_a_quantizer,
                                        cast(Quantizer, submodule.selected_w_quantizer),
                                        b_quantizer)
        # per-channel search => multiple precisions/quantizers
        elif isinstance(submodule.w_mps_quantizer, MPSPerChannelQtz):
            selected_w_precision = cast(List[int], submodule.selected_w_precision)
            selected_w_quantizer = cast(List[Quantizer], submodule.selected_w_quantizer)
            nn_list = []
            prec_and_quantz = dict(zip(selected_w_precision, selected_w_quantizer))
            # TODO: debug this. Isn't it doing multiple iterations on the same precision?
            for prec, w_quant in prec_and_quantz.items():
                mask = [c == prec for c in selected_w_precision]
                out_channels = sum(mask)
                if out_channels == 0:  # no out_channels for the current prec
                    continue
                new_conv = nn.Conv2d(submodule.in_channels,
                                     out_channels,
                                     submodule.kernel_size,
                                     submodule.stride,
                                     submodule.padding,
                                     submodule.dilation,
                                     submodule.groups,
                                     submodule.bias is not None,
                                     submodule.padding_mode)
                new_weights = submodule.weight[mask, :, :, :]
                with torch.no_grad():
                    new_conv.weight.copy_(new_weights)
                    if submodule.bias is not None:
                        cast(nn.parameter.Parameter, new_conv.bias).copy_(submodule.bias[mask])
                        # re-create bias quantizer using correct number of channels
                        # TODO: DP: shouldn't we also recreate the w_quantizer with fewer channels?
                        b_quantizer_class = submodule.b_mps_quantizer.quantizer
                        b_quantizer_kwargs = submodule.b_mps_quantizer.quantizer_kwargs
                        b_quantizer_kwargs['cout'] = out_channels
                        b_quantizer = b_quantizer_class(**b_quantizer_kwargs)
                    else:
                        b_quantizer = None
                quant_conv = QuantConv2d(new_conv,
                                         submodule.selected_in_a_quantizer,
                                         submodule.selected_out_a_quantizer,
                                         w_quant,
                                         b_quantizer)
                nn_list.append(quant_conv)
            new_submodule = QuantList(nn_list)
        else:
            msg = f'Supported mixed-precision types: {list(MPSType)}'
            raise ValueError(msg)
        mod.add_submodule(str(n.target), new_submodule)

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer hyperparameters

        :return: a dictionary containing the optimized layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        # TODO: name incompatibility with get_modified_vars
        return {
            'in_a_precision': self.selected_in_a_precision,
            'out_a_precision': self.selected_out_a_precision,
            'w_precision': self.selected_w_precision,
        }

    def get_modified_vars(self) -> Iterator[Dict[str, Any]]:
        """Method that returns the modified vars(self) dictionary for the instance, for each
        combination of supported precision, used for cost computation

        :return: an iterator over the modified vars(self) data structures
        :rtype: Iterator[Dict[str, Any]]
        """
        # TODO: check this function
        for i, a_prec in enumerate(self.in_a_mps_quantizer.precisions):
            for j, w_prec in enumerate(self.w_mps_quantizer.precisions):
                v = dict(vars(self))
                v['in_bits'] = a_prec
                v['in_format'] = int
                v['w_bits'] = w_prec
                v['w_format'] = int
                # downscale the input_channels times the probability of using that
                # input precision
                # TODO: detach added based on Beatrice and Alessio's observations on back-prop.
                # To be double-checked
                v['in_channels'] = (self.input_features_calculator.features.detach() *
                                    self.in_a_mps_quantizer.theta_alpha[i])
                # same with weights precision and output channels, but distinguish the two types
                # of quantizer
                if isinstance(self.w_mps_quantizer, MPSPerLayerQtz):
                    v['out_channels'] = (self.out_channels *
                                         self.w_mps_quantizer.theta_alpha[j])
                elif isinstance(self.w_mps_quantizer, MPSPerChannelQtz):
                    v['out_channels'] = self.w_mps_quantizer.theta_alpha[j, :].sum()
                else:
                    msg = f'Supported mixed-precision types: {list(MPSType)}'
                    raise ValueError(msg)
                yield v

    def named_nas_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the architectural parameters of this layer, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: kept for uniformity with pytorch API,
        but MixPrecModule never have sub-layers TODO: check if true
        :type recurse: bool
        :return: an iterator over the architectural parameters of this layer
        :rtype: Iterator[nn.Parameter]
        """
        prfx = prefix
        prfx += "." if len(prefix) > 0 else ""
        for name, param in self.out_a_mps_quantizer.named_parameters(
                prfx + "out_a_mps_quantizer", recurse):
            yield name, param
        for name, param in self.w_mps_quantizer.named_parameters(
                prfx + "w_mps_quantizer", recurse):
            yield name, param
        # no bias MPS quantizer since it is sharing the parameters of the act and weights

    @property
    def selected_in_a_precision(self) -> int:
        """Return the selected precision based on the magnitude of `alpha_prec`
        components for input activations

        :return: the selected precision
        :rtype: int
        """
        with torch.no_grad():
            idx = int(torch.argmax(self.in_a_mps_quantizer.alpha_prec))
            return int(self.in_a_mps_quantizer.precisions[idx])

    @property
    def selected_out_a_precision(self) -> Union[int, str]:
        """Return the selected precision based on the magnitude of `alpha_prec`
        components for output activations.
        If output is not quantized returns the 'float' string.

        :return: the selected precision
        :rtype: Union[int, str]
        """
        if type(self.out_a_mps_quantizer) != nn.Identity:
            with torch.no_grad():
                idx = int(torch.argmax(self.out_a_mps_quantizer.alpha_prec))
                return int(self.out_a_mps_quantizer.precisions[idx])
        else:
            return 'float'

    @property
    def selected_w_precision(self) -> Union[int, List[int]]:
        """Return the selected precision(s) based on the magnitude of `alpha_prec`
        components for weights

        :return: the selected precision(s)
        :rtype: Union[int, List[int]]
        """
        with torch.no_grad():
            if isinstance(self.w_mps_quantizer, MPSPerLayerQtz):
                idx = int(torch.argmax(self.w_mps_quantizer.alpha_prec))
                return int(self.w_mps_quantizer.precisions[idx])
            elif isinstance(self.w_mps_quantizer, MPSPerChannelQtz):
                idx = torch.argmax(self.w_mps_quantizer.alpha_prec, dim=0)
                return [int(self.w_mps_quantizer.precisions[int(i)]) for i in idx]
            else:
                msg = f'Supported mixed-precision types: {list(MPSType)}'
                raise ValueError(msg)

    @property
    def selected_in_a_quantizer(self) -> Quantizer:
        """Return the selected quantizer based on the magnitude of `alpha_prec`
        components for input activations

        :return: the selected quantizer
        :rtype: Quantizer
        """
        with torch.no_grad():
            idx = int(torch.argmax(self.in_a_mps_quantizer.alpha_prec))
            qtz = self.in_a_mps_quantizer.qtz_funcs[idx]
            qtz = cast(Quantizer, qtz)
            return qtz

    @property
    def selected_out_a_quantizer(self) -> Quantizer:
        """Return the selected quantizer based on the magnitude of `alpha_prec`
        components for output activations

        :return: the selected quantizer
        :rtype: Quantizer
        """
        if type(self.out_a_mps_quantizer) != nn.Identity:
            with torch.no_grad():
                idx = int(torch.argmax(self.out_a_mps_quantizer.alpha_prec))
                qtz = self.out_a_mps_quantizer.qtz_funcs[idx]
                qtz = cast(Quantizer, qtz)
                return qtz
        else:
            # TODO: DP: when is this used? Output layer?
            qtz = cast(Quantizer, self.out_a_mps_quantizer)
            return qtz

    @property
    def selected_w_quantizer(self) -> Union[Quantizer, List[Quantizer]]:
        """Return the selected quantizer(s) based on the magnitude of `alpha_prec`
        components for weights

        :return: the selected quantizer(s)
        :rtype: Union[Quantizer, List[Quantizer]]
        """
        with torch.no_grad():
            if isinstance(self.w_mps_quantizer, MPSPerLayerQtz):
                idx = int(torch.argmax(self.w_mps_quantizer.alpha_prec))
                qtz = self.w_mps_quantizer.qtz_funcs[idx]
                qtz = cast(Quantizer, qtz)
                return qtz
            elif isinstance(self.w_mps_quantizer, MPSPerChannelQtz):
                idx = torch.argmax(self.w_mps_quantizer.alpha_prec, dim=0)
                qtz = [self.w_mps_quantizer.qtz_funcs[i] for i in idx]
                qtz = cast(List[Quantizer], qtz)
                return qtz
            else:
                msg = f'Supported mixed-precision types: {list(MPSType)}'
                raise ValueError(msg)

    @property
    def out_features_eff(self) -> torch.Tensor:
        """Returns the number of not pruned channels for this layer.

        :return: the number of not pruned channels for this layer.
        :rtype: torch.Tensor
        """
        if isinstance(self.w_mps_quantizer, MPSPerChannelQtz):
            return cast(torch.Tensor, self.w_mps_quantizer.out_features_eff)
        else:
            return cast(torch.Tensor, self.out_channels)

    @property
    def number_pruned_channels(self) -> torch.Tensor:
        """Returns the number of pruned channels for this layer.

        :return: the number of pruned channels for this layer.
        :rtype: torch.Tensor
        """
        return self.out_channels - self.out_features_eff

    @property
    def input_features_calculator(self) -> FeaturesCalculator:
        """Returns the `FeaturesCalculator` instance that computes the number of input features for
        this layer.

        :return: the `FeaturesCalculator` instance that computes the number of input features for
        this layer.
        :rtype: FeaturesCalculator
        """
        return self._input_features_calculator

    @input_features_calculator.setter
    def input_features_calculator(self, calc: FeaturesCalculator):
        """Set the `FeaturesCalculator` instance that computes the number of input features for
        this layer.

        :param calc: the `FeaturesCalculator` instance that computes the number of input features
        for this layer
        :type calc: FeaturesCalculator
        """
        calc.register(self)
        self._input_features_calculator = calc

    @property
    def in_a_mps_quantizer(self) -> MPSPerLayerQtz:
        """Returns the `MPSQtzLayer` for input activations calculation

        :return: the `MPSQtzLayer` instance that computes mixprec quantized
        versions of the input activations
        :rtype: MPSQtzLayer
        """
        return self._in_a_mps_quantizer

    @in_a_mps_quantizer.setter
    def in_a_mps_quantizer(self, qtz: MPSPerLayerQtz):
        """Set the `MPSQtzLayer` for input activations calculation

        :param qtz: the `MPSQtzLayer` instance that computes mixprec quantized
        versions of the input activations
        :type qtz: MPSQtzLayer
        """
        self._in_a_mps_quantizer = qtz
        self.b_mps_quantizer.in_a_mps_quantizer = self._in_a_mps_quantizer