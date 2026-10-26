import pytest
import torch

from models.modules.moe_dict import MoEDictionary


def test_expert_banks_and_projections_have_expected_shapes_without_copying_atoms():
    model = MoEDictionary(d=64, m=256, num_experts=4)

    banks = model.expert_dictionaries()
    keys, values = model.project_experts()

    assert banks.shape == (4, 64, 64)
    assert keys.shape == (4, 64, 16)
    assert values.shape == (4, 64, 64)
    assert model.dictionary_parameter_count == 256 * 64
    assert banks.untyped_storage().data_ptr() == model.dictionary.untyped_storage().data_ptr()


def test_dynamic_residual_refines_banks_and_backpropagates():
    model = MoEDictionary(d=8, m=16, num_experts=4, reduction_ratio=2)
    residual = torch.randn(4, 4, 8, requires_grad=True)

    keys, values = model.project_experts(residual)
    (keys.square().mean() + values.square().mean()).backward()

    assert residual.grad is not None
    assert torch.count_nonzero(residual.grad).item() > 0
    assert model.dictionary.grad is not None
    assert torch.count_nonzero(model.dictionary.grad).item() > 0
    assert model.dictionary_parameter_count == 16 * 8


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"m": 10, "num_experts": 4}, "must be divisible"),
        ({"num_experts": 0}, "num_experts must be positive"),
        ({"d": 10, "reduction_ratio": 4}, "must be divisible"),
    ],
)
def test_rejects_invalid_dictionary_geometry(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MoEDictionary(**kwargs)


def test_rejects_residual_with_wrong_shape():
    model = MoEDictionary(d=8, m=16, num_experts=4)
    with pytest.raises(ValueError, match="expected residual shape"):
        model.expert_dictionaries(torch.zeros(4, 3, 8))
