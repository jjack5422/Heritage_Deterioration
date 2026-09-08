from dual_adapter_sam3.model import DualAdapterSam3, DualAdapterSam3Output


def test_model_name_describes_the_architecture_only() -> None:
    assert DualAdapterSam3.__name__ == "DualAdapterSam3"
    assert DualAdapterSam3Output.__name__ == "DualAdapterSam3Output"
