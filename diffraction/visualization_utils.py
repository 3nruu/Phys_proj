import numpy as np

def compute_smart_color_limits(data, percentile_high=99.5, percentile_low=1):
    """
    Вычисляет разумные пределы цветовой шкалы, игнорируя выбросы.
    
    Args:
        data: 2D массив данных
        percentile_high: верхний перцентиль (по умолчанию 99.5)
        percentile_low: нижний перцентиль (по умолчанию 1)
    
    Returns:
        (vmin, vmax) - рекомендуемые пределы
    """
    data_flat = data.flatten()
    data_positive = data_flat[data_flat > 0]
    
    if len(data_positive) == 0:
        return 0, 1
    
    vmin = np.percentile(data_positive, percentile_low)
    vmax = np.percentile(data_positive, percentile_high)
    
    # Защита от слишком узкого диапазона
    if vmax - vmin < 0.01 * np.max(data_positive):
        vmin = 0
        vmax = np.max(data_positive)
    
    return float(vmin), float(vmax)

def normalize_for_display(data, vmin=None, vmax=None):
    """
    Нормализует данные для отображения.
    """
    if vmin is None or vmax is None:
        vmin, vmax = compute_smart_color_limits(data)
    
    # Масштабируем и обрезаем
    scaled = (data - vmin) / (vmax - vmin + 1e-12)
    scaled = np.clip(scaled, 0, 1)
    return scaled, vmin, vmax