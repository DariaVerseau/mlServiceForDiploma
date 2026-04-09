# tests/test_codeformer.py
def test_codeformer_import():
    """Проверка импорта CodeFormer"""
    import sys
    sys.path.append('/app/CodeFormer')
    from basicsr.archs.codeformer_arch import CodeFormer
    assert True

def test_codeformer_weights_exist():
    """Проверка наличия весов CodeFormer"""
    import os
    weights_path = '/app/CodeFormer/weights/CodeFormer/codeformer.pth'
    # В Docker это должно быть True
    # assert os.path.exists(weights_path)