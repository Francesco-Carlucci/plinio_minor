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

from typing import Dict, Any, Optional, Iterator, Tuple, Type, cast, Union, List
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from ..quant.quantizers import Quantizer
from ..quant.nn import Quant_Linear, Quant_List
from .mixprec_module import MixPrecModule
from .mixprec_qtz import MixPrecType, MixPrec_Qtz_Layer, MixPrec_Qtz_Channel, \
    MixPrec_Qtz_Layer_Bias, MixPrec_Qtz_Channel_Bias


class MixPrec_Linear(nn.Linear, MixPrecModule):
    """A nn.Module implementing a Linear layer with mixed-precision search support

    :param linear: the inner `nn.Linear` layer to be optimized
    :type linear: nn.Linear
    :param a_precisions: different bitwitdth alternatives among which perform search
    for activations
    :type a_precisions: Tuple[int, ...]
    :param w_precisions: different bitwitdth alternatives among which perform search
    for weights
    :type w_precisions: Tuple[int, ...]
    :param a_quantizer: activation quantizer
    :type a_quantizer: MixPrec_Qtz_Layer
    :param w_quantizer: weight quantizer
    :type w_quantizer: Union[MixPrec_Qtz_Layer, MixPrec_Qtz_Channel]
    :param b_quantizer: bias quantizer
    :type b_quantizer: Union[MixPrec_Qtz_Layer, MixPrec_Qtz_Channel]
    :param w_mixprec_type: the mixed precision strategy to be used for weigth
    i.e., `PER_CHANNEL` or `PER_LAYER`.
    :type w_mixprec_type: MixPrecType
    """
    def __init__(self,
                 linear: nn.Linear,
                 a_precisions: Tuple[int, ...],
                 w_precisions: Tuple[int, ...],
                 a_quantizer: MixPrec_Qtz_Layer,
                 w_quantizer: Union[MixPrec_Qtz_Layer, MixPrec_Qtz_Channel],
                 b_quantizer: Union[MixPrec_Qtz_Layer, MixPrec_Qtz_Channel],
                 w_mixprec_type: MixPrecType):
        super(MixPrec_Linear, self).__init__(
            linear.in_features,
            linear.out_features,
            linear.bias is not None)
        with torch.no_grad():
            self.weight.copy_(linear.weight)
            if linear.bias is not None:
                self.bias = cast(torch.nn.parameter.Parameter, self.bias)
                self.bias.copy_(linear.bias)
            else:
                self.bias = None

        self.a_precisions = a_precisions
        self.w_precisions = w_precisions
        self.mixprec_a_quantizer = a_quantizer
        self.mixprec_w_quantizer = w_quantizer
        if self.bias is not None:
            self.mixprec_b_quantizer = b_quantizer
            # Share NAS parameters of weights and biases
            # self.mixprec_b_quantizer.alpha_prec = self.mixprec_w_quantizer.alpha_prec
        else:
            self.mixprec_b_quantizer = lambda *args: None  # Do Nothing

        self.w_mixprec_type = w_mixprec_type

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the mixed-precision NAS-able layer.

        In a nutshell:
        - Quantize and combine the input tensor at the different `precisions`.
        - Quantize and combine the weight tensor at the different `precisions`.
        - Compute Linear operation using the previous obtained quantized tensors.

        :param input: the input activations tensor
        :type input: torch.Tensor
        :return: the output activations tensor
        :rtype: torch.Tensor
        """
        # Quantization
        q_inp = self.mixprec_a_quantizer(input)
        q_w = self.mixprec_w_quantizer(self.weight)
        q_b = self.mixprec_b_quantizer(self.bias)

        # Linear operation
        q_out = F.linear(q_inp, q_w, q_b)

        # TODO: Understand what I need to do for regularization
        # Save info for regularization

        return q_out

    @staticmethod
    def autoimport(n: fx.Node,
                   mod: fx.GraphModule,
                   w_mixprec_type: MixPrecType,
                   a_precisions: Tuple[int, ...],
                   w_precisions: Tuple[int, ...],
                   a_quantizer: Type[Quantizer],
                   w_quantizer: Type[Quantizer],
                   b_quantizer: Type[Quantizer],
                   a_sq: Optional[Quantizer],
                   a_quantizer_kwargs: Dict = {},
                   w_quantizer_kwargs: Dict = {},
                   b_quantizer_kwargs: Dict = {}
                   ) -> Optional[Quantizer]:
        """Create a new fx.Node relative to a MixPrec_Linear layer, starting from the fx.Node
        of a nn.Linear layer, and replace it into the parent fx.GraphModule

        Also returns a quantizer in case it needs to be shared with other layers

        :param n: a fx.Node corresponding to a nn.ReLU layer, with shape annotations
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param w_mixprec_type: the mixed precision strategy to be used for weigth
        i.e., `PER_CHANNEL` or `PER_LAYER`.
        :type w_mixprec_type: MixPrecType
        :param a_precisions: The precisions to be explored for activations
        :type a_precisions: Tuple[int, ...]
        :param w_precisions: The precisions to be explored for weights
        :type w_precisions: Tuple[int, ...]
        :param a_quantizer: The quantizer to be used for activations
        :type a_quantizer: Type[Quantizer]
        :param w_quantizer: The quantizer to be used for weights
        :type w_quantizer: Type[Quantizer]
        :param b_quantizer: The quantizer to be used for biases
        :type b_quantizer: Type[Quantizer]
        :param a_sq: An optional shared quantizer derived from other layers for activations
        :type a_sq: Optional[Quantizer]
        :param a_quantizer_kwargs: act quantizer kwargs, if no kwargs are passed default is used
        :type a_quantizer_kwargs: Dict
        :param w_quantizer_kwargs: weight quantizer kwargs, if no kwargs are passed default is used
        :type w_quantizer_kwargs: Dict
        :param b_quantizer_kwargs: bias quantizer kwargs, if no kwargs are passed default is used
        :type b_quantizer_kwargs: Dict
        :raises TypeError: if the input fx.Node is not of the correct type
        :return: the updated shared quantizer
        :rtype: Optional[Quantizer]
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != nn.Linear:
            msg = f"Trying to generate MixPrec_Linear from layer of type {type(submodule)}"
            raise TypeError(msg)
        # here, this is guaranteed
        submodule = cast(nn.Linear, submodule)
        # Build activation mixprec quantizer
        if a_sq is not None:
            mixprec_a_quantizer = a_sq
        else:
            mixprec_a_quantizer = MixPrec_Qtz_Layer(a_precisions,
                                                    a_quantizer,
                                                    a_quantizer_kwargs)
        # Build weight mixprec quantizer
        if w_mixprec_type == MixPrecType.PER_LAYER:
            mixprec_w_quantizer = MixPrec_Qtz_Layer(w_precisions,
                                                    w_quantizer,
                                                    w_quantizer_kwargs)
        elif w_mixprec_type == MixPrecType.PER_CHANNEL:
            mixprec_w_quantizer = MixPrec_Qtz_Channel(w_precisions,
                                                      submodule.out_features,
                                                      w_quantizer,
                                                      w_quantizer_kwargs)
        else:
            msg = f'Supported mixed-precision types: {list(MixPrecType)}'
            raise ValueError(msg)

        # Build bias mixprec quantizer
        b_mixprec_type = w_mixprec_type  # Bias MixPrec scheme is dictated by weights
        if b_mixprec_type == MixPrecType.PER_LAYER:
            mixprec_a_quantizer = cast(MixPrec_Qtz_Layer, mixprec_a_quantizer)
            mixprec_w_quantizer = cast(MixPrec_Qtz_Layer, mixprec_w_quantizer)
            mixprec_b_quantizer = MixPrec_Qtz_Layer_Bias(b_quantizer,
                                                         submodule.out_features,
                                                         mixprec_a_quantizer,
                                                         mixprec_w_quantizer,
                                                         b_quantizer_kwargs)
        elif w_mixprec_type == MixPrecType.PER_CHANNEL:
            mixprec_a_quantizer = cast(MixPrec_Qtz_Layer, mixprec_a_quantizer)
            mixprec_w_quantizer = cast(MixPrec_Qtz_Channel, mixprec_w_quantizer)
            mixprec_b_quantizer = MixPrec_Qtz_Channel_Bias(b_quantizer,
                                                           submodule.out_features,
                                                           mixprec_a_quantizer,
                                                           mixprec_w_quantizer,
                                                           b_quantizer_kwargs)
        else:
            msg = f'Supported mixed-precision types: {list(MixPrecType)}'
            raise ValueError(msg)

        mixprec_a_quantizer = cast(MixPrec_Qtz_Layer, mixprec_a_quantizer)
        mixprec_w_quantizer = cast(Union[MixPrec_Qtz_Layer, MixPrec_Qtz_Channel],
                                   mixprec_w_quantizer)
        mixprec_b_quantizer = cast(Union[MixPrec_Qtz_Layer, MixPrec_Qtz_Channel],
                                   mixprec_b_quantizer)
        new_submodule = MixPrec_Linear(submodule,
                                       a_precisions,
                                       w_precisions,
                                       mixprec_a_quantizer,
                                       mixprec_w_quantizer,
                                       mixprec_b_quantizer,
                                       w_mixprec_type)
        mod.add_submodule(str(n.target), new_submodule)
        return None  # TODO: Understand if I should return something and when

    @staticmethod
    def export(n: fx.Node, mod: fx.GraphModule):
        """Replaces a fx.Node corresponding to a MixPrec_Linear layer,
        with the selected fake-quantized nn.Linear layer within a fx.GraphModule

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != MixPrec_Linear:
            raise TypeError(f"Trying to export a layer of type {type(submodule)}")

        # Select precision and quantizer for activations
        selected_a_precision = submodule.selected_a_precision
        selected_a_precision = cast(int, selected_a_precision)
        selected_a_quantizer = submodule.selected_a_quantizer
        selected_a_quantizer = cast(Type[Quantizer], selected_a_quantizer)

        # Select precision(s) and quantizer(s) for weights and biases
        selected_w_precision = submodule.selected_w_precision
        selected_w_quantizer = submodule.selected_w_quantizer
        # w_mixprec_type is `PER_LAYER` => single precision/quantizer
        if submodule.w_mixprec_type == MixPrecType.PER_LAYER:
            selected_w_precision = cast(int, selected_w_precision)
            selected_w_quantizer = cast(Type[Quantizer], selected_w_quantizer)
            if submodule.bias is not None:
                submodule.mixprec_b_quantizer = cast(MixPrec_Qtz_Layer_Bias,
                                                     submodule.mixprec_b_quantizer)
                # Build bias quantizer using s_factors corresponding to selected
                # act and weights quantizers
                b_quantizer_class = submodule.mixprec_b_quantizer.quantizer
                b_quantizer_class = cast(Type[Quantizer], b_quantizer_class)
                b_quantizer_kwargs = submodule.mixprec_b_quantizer.quantizer_kwargs
                b_quantizer = b_quantizer_class(**b_quantizer_kwargs)
                b_quantizer = cast(Type[Quantizer], b_quantizer)
            else:
                b_quantizer = None
            submodule = cast(nn.Linear, submodule)
            new_submodule = Quant_Linear(submodule,
                                         selected_a_precision,
                                         selected_w_precision,
                                         selected_a_quantizer,
                                         selected_w_quantizer,
                                         b_quantizer)
        # w_mixprec_type is `PER_CHANNEL` => multiple precision/quantizer
        elif submodule.w_mixprec_type == MixPrecType.PER_CHANNEL:
            selected_w_precision = cast(List[int], selected_w_precision)
            selected_w_quantizer = cast(List[Type[Quantizer]], selected_w_quantizer)
            submodule = cast(nn.Linear, submodule)
            nn_list = []
            prec_and_quantz = dict(zip(selected_w_precision, selected_w_quantizer))
            for prec, w_quant in prec_and_quantz.items():
                mask = [c == prec for c in selected_w_precision]
                out_features = sum(mask)
                if out_features == 0:  # No out_features for the current prec
                    continue
                new_lin = nn.Linear(submodule.in_features,
                                    out_features,
                                    submodule.bias is not None)
                new_weights = submodule.weight[mask, :]
                with torch.no_grad():
                    new_lin.weight.copy_(new_weights)
                    if submodule.bias is not None:
                        new_lin.bias.copy_(submodule.bias[mask])
                        submodule.mixprec_b_quantizer = cast(MixPrec_Qtz_Channel_Bias,
                                                             submodule.mixprec_b_quantizer)
                        # Build bias quantizer using s_factors corresponding to selected
                        # act and weights quantizers
                        b_quantizer_class = submodule.mixprec_b_quantizer.quantizer
                        b_quantizer_class = cast(Type[Quantizer], b_quantizer_class)
                        b_quantizer_kwargs = submodule.mixprec_b_quantizer.quantizer_kwargs
                        b_quantizer_kwargs['cout'] = out_features
                        b_quantizer = b_quantizer_class(**b_quantizer_kwargs)
                        b_quantizer = cast(Type[Quantizer], b_quantizer)
                    else:
                        b_quantizer = None
                quant_lin = Quant_Linear(new_lin,
                                         selected_a_precision,
                                         prec,
                                         selected_a_quantizer,
                                         w_quant,
                                         b_quantizer)
                nn_list.append(quant_lin)
            new_submodule = Quant_List(nn_list)
        else:
            msg = f'Supported mixed-precision types: {list(MixPrecType)}'
            raise ValueError(msg)

        mod.add_submodule(str(n.target), new_submodule)

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer hyperparameters

        :return: a dictionary containing the optimized layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            'a_precision': self.selected_a_precision,
            'w_precision': self.selected_w_precision,
        }

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
        for name, param in self.mixprec_a_quantizer.named_parameters(
                prfx + "mixprec_a_quantizer", recurse):
            yield name, param
        for name, param in self.mixprec_w_quantizer.named_parameters(
                prfx + "mixprec_w_quantizer", recurse):
            yield name, param
        for name, param in self.mixprec_b_quantizer.named_parameters(
                prfx + "mixprec_b_quantizer", recurse):
            yield name, param

    @property
    def selected_a_precision(self) -> int:
        """Return the selected precision based on the magnitude of `alpha_prec`
        components for activations

        :return: the selected precision
        :rtype: int
        """
        with torch.no_grad():
            idx = int(torch.argmax(self.mixprec_a_quantizer.alpha_prec))
            return self.a_precisions[idx]

    @property
    def selected_w_precision(self) -> Union[int, List[int]]:
        """Return the selected precision(s) based on the magnitude of `alpha_prec`
        components for weights

        :return: a function returning the selected precision(s)
        :rtype: Union[int, List[int]]
        """
        with torch.no_grad():
            if self.w_mixprec_type == MixPrecType.PER_LAYER:
                idx = int(torch.argmax(self.mixprec_w_quantizer.alpha_prec))
                return self.w_precisions[idx]
            elif self.w_mixprec_type == MixPrecType.PER_CHANNEL:
                idx = torch.argmax(self.mixprec_w_quantizer.alpha_prec, dim=0)
                return [self.w_precisions[int(i)] for i in idx]
            else:
                msg = f'Supported mixed-precision types: {list(MixPrecType)}'
                raise ValueError(msg)

    @property
    def selected_a_quantizer(self) -> Type[Quantizer]:
        """Return the selected quantizer based on the magnitude of `alpha_prec`
        components for activations

        :return: the selected quantizer(s)
        :rtype: Type[Quantizer]
        """
        with torch.no_grad():
            idx = int(torch.argmax(self.mixprec_a_quantizer.alpha_prec))
            qtz = self.mixprec_a_quantizer.mix_qtz[idx]
            qtz = cast(Type[Quantizer], qtz)
            return qtz

    @property
    def selected_w_quantizer(self) -> Union[Type[Quantizer], List[Type[Quantizer]]]:
        """Return the selected quantizer(s) based on the magnitude of `alpha_prec`
        components for weights

        :return: the selected quantizer(s)
        :rtype: Union[Type[Quantizer], List[Type[Quantizer]]]
        """
        with torch.no_grad():
            if self.w_mixprec_type == MixPrecType.PER_LAYER:
                idx = int(torch.argmax(self.mixprec_w_quantizer.alpha_prec))
                qtz = self.mixprec_w_quantizer.mix_qtz[idx]
                qtz = cast(Type[Quantizer], qtz)
                return qtz
            elif self.w_mixprec_type == MixPrecType.PER_CHANNEL:
                idx = torch.argmax(self.mixprec_w_quantizer.alpha_prec, dim=0)
                qtz = [self.mixprec_w_quantizer.mix_qtz[i] for i in idx]
                qtz = cast(List[Type[Quantizer]], qtz)
                return qtz
            else:
                msg = f'Supported mixed-precision types: {list(MixPrecType)}'
                raise ValueError(msg)

    @property
    def selected_b_quantizer(self) -> Type[Quantizer]:
        """Return the selected quantizer based on the magnitude of `alpha_prec`
        components for biases

        :return: the selected quantizer
        :rtype: Type[Quantizer]
        """
        with torch.no_grad():
            idx = int(torch.argmax(self.mixprec_b_quantizer.alpha_prec))
            qtz = self.mixprec_b_quantizer.mix_qtz[idx]
            qtz = cast(Type[Quantizer], qtz)
            return qtz

    def get_size(self) -> torch.Tensor:
        """Computes the effective number of weights for the layer

        :return: the effective memory occupation of weights
        :rtype: torch.Tensor
        """
        eff_w_prec = self.mixprec_w_quantizer.effective_precision
        cost = self.in_features * self.out_features * eff_w_prec
        return cost

    # N.B., EdMIPS formulation
    def get_macs(self) -> torch.Tensor:
        """Method that computes the effective MACs operations for the layer

        :return: the effective number of MACs
        :rtype: torch.Tensor
        """
        eff_w_prec = self.mixprec_w_quantizer.effective_precision
        eff_a_prec = self.mixprec_a_quantizer.effective_precision
        cost = self.in_features * self.out_features * eff_w_prec * eff_a_prec
        return cost