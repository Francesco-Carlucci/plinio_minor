import unittest
import torch
from flexnas.methods.supernet.supernet import SuperNet
from unit_test.models.supernet_nn import SingleModuleNet1, SingleModuleNet2
from unit_test.models.supernet_nn import MultipleModuleNet1, StandardSNModule


class TestSuperNet(unittest.TestCase):

    # Single Module
    def test_supernet_singleModule_output_shape(self):
        batch_size = 1
        in_length = 4
        out_length = 4

        ch_in = 32
        ch_out = 57
        model = SingleModuleNet1()
        sn_model = SuperNet(model, (ch_in, in_length))
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_length))
        out = sn_model(dummy_inp)
        self.assertEqual(out.shape, (batch_size, ch_out, out_length), "Unexpected output shape")

        ch_in = 32
        ch_out = 32
        model = SingleModuleNet2()
        sn_model = SuperNet(model, (ch_in, in_length))
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_length))
        out = sn_model(dummy_inp)
        self.assertEqual(out.shape, (batch_size, ch_out, out_length), "Unexpected output shape")

    def test_supernet_singleModule_target_modules(self):
        ch_in = 32
        in_length = 4
        n_target_modules = 1

        model = SingleModuleNet1()
        sn_model = SuperNet(model, (ch_in, in_length))
        target_modules = sn_model._target_modules
        self.assertEqual(len(target_modules), 1, "Wrong target modules number")

        model = SingleModuleNet2()
        sn_model = SuperNet(model, (ch_in, in_length))
        target_modules = sn_model._target_modules
        self.assertEqual(len(target_modules), n_target_modules, "Wrong target modules number")

    def test_supernet_singleModule_input_shape(self):
        ch_in = 32
        in_length = 4
        batch_size = 1

        model = SingleModuleNet1()
        sn_model = SuperNet(model, (ch_in, in_length))
        target_modules = sn_model._target_modules
        shapes = [(batch_size, ch_in, in_length)]
        self.assertEqual(target_modules[0][1].input_shape, shapes[0], "Wrong target input shapes")

        model = SingleModuleNet2()
        sn_model = SuperNet(model, (ch_in, in_length))
        target_modules = sn_model._target_modules
        shapes = [(batch_size, ch_in, in_length)]
        self.assertEqual(target_modules[0][1].input_shape, shapes[0], "Wrong target input shapes")

    def test_supernet_singleModule_size(self):
        ch_in = 32
        in_length = 4

        model = SingleModuleNet1()
        sn_model = SuperNet(model, (ch_in, in_length))
        self.assertEqual(sn_model.get_size(), 9177)

        model = SingleModuleNet2()
        sn_model = SuperNet(model, (ch_in, in_length))
        self.assertEqual(sn_model.get_size(), 1552)

    def test_supernet_singleModule_macs(self):
        ch_in = 32
        in_length = 4

        model = SingleModuleNet1()
        sn_model = SuperNet(model, (ch_in, in_length))
        self.assertEqual(sn_model.get_macs(), 36708)

        model = SingleModuleNet2()
        sn_model = SuperNet(model, (ch_in, in_length))
        self.assertEqual(sn_model.get_macs(), 6208)

    # Multiple Modules
    def test_supernet_multipleModule_output_shape(self):
        batch_size = 1
        in_length = 4
        out_length = 4

        ch_in = 32
        ch_out = 57
        model = MultipleModuleNet1()
        sn_model = SuperNet(model, (ch_in, in_length))
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_length))
        out = sn_model(dummy_inp)
        self.assertEqual(out.shape, (batch_size, ch_out, out_length), "Unexpected output shape")

    def test_supernet_multipleModule_target_modules(self):
        ch_in = 32
        in_length = 4
        n_target_modules = 2

        model = MultipleModuleNet1()
        sn_model = SuperNet(model, (ch_in, in_length))
        target_modules = sn_model._target_modules
        self.assertEqual(len(target_modules), n_target_modules, "Wrong target modules number")

    def test_supernet_multipleModule_input_shape(self):
        ch_in = 32
        in_length = 4
        batch_size = 1

        model = MultipleModuleNet1()
        sn_model = SuperNet(model, (ch_in, in_length))
        target_modules = sn_model._target_modules
        shapes = [(batch_size, ch_in, in_length), (batch_size, ch_in, in_length)]
        for i, t in enumerate(target_modules):
            self.assertEqual(t[1].input_shape, shapes[i], "Wrong input target input shapes")

    def test_supernet_multipleModule_size(self):
        ch_in = 32
        in_length = 4

        model = MultipleModuleNet1()
        sn_model = SuperNet(model, (ch_in, in_length))
        self.assertEqual(sn_model.get_size(), 10729)

    def test_supernet_multipleModule_macs(self):
        ch_in = 32
        in_length = 4

        model = MultipleModuleNet1()
        sn_model = SuperNet(model, (ch_in, in_length))
        self.assertEqual(sn_model.get_macs(), 42916)

    def test_standardSNModule(self):
        ch_in = 32
        ch_out = 32
        in_width = 64
        in_height = 64
        out_width = 64
        out_heigth = 64
        batch_size = 1

        model = StandardSNModule()
        sn_model = SuperNet(model, (ch_in, in_width, in_height))
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out = sn_model(dummy_inp)
        self.assertEqual(out.shape, (batch_size, ch_out, out_width, out_heigth),
                         "Unexpected output shape")


if __name__ == '__main__':
    unittest.main(verbosity=2)