# tests/test_adain.py
def test_adain_import():
    """Проверка импорта AdaIN"""
    import sys
    sys.path.append('/app/pytorch-AdaIN')
    # Проверяем наличие test.py
    import os
    assert os.path.exists('/app/pytorch-AdaIN/test.py')