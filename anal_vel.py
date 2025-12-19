#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# -------------------------------------------------
# Загрузка данных
# -------------------------------------------------
filename = "QM_vs_Fresnel_Comparison.pkl"

with open(filename, "rb") as f:
    data = pickle.load(f)

# Извлекаем данные
params = data['params']
y = data['y']
x = data['x']
qm_best = data['qm_slice_best']
fresnel_best = data['fresnel_slice_best']
qm_slices_fixed = data['qm_slices_fixed']
t_vals = data['t_vals']
I_fresnel_spacetime = data['I_fresnel_spacetime']
z_vals = data['z_vals']

# Метрики
error_L1 = data['error_L1']
error_L2 = data['error_L2']
correlation = data['correlation']
t_best = data['t_best']

# -------------------------------------------------
# Нормировка для сравнения форм
# -------------------------------------------------
qm_norm = qm_best / (qm_best.max() + 1e-15)
fresnel_norm = fresnel_best / (fresnel_best.max() + 1e-15)

# -------------------------------------------------
# Построение графиков
# -------------------------------------------------
plt.rcParams.update({'font.size': 10})
fig = plt.figure(figsize=(15, 10))
gs = plt.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# 1. Сравнение профилей на экране
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(y, qm_norm, 'b-', lw=2.5, label='QM (нормировано)')
ax1.plot(y, fresnel_norm, 'r--', lw=2.5, label='Френель (нормировано)')
ax1.set_xlabel('y')
ax1.set_ylabel('Нормированная интенсивность')
ax1.set_title('Сравнение профилей на экране')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Разность профилей
ax2 = fig.add_subplot(gs[0, 2])
diff_1d = np.abs(qm_norm - fresnel_norm)
ax2.plot(y, diff_1d, 'k-', lw=2)
ax2.set_xlabel('y')
ax2.set_ylabel('|QM - Френель|')
ax2.set_title('Абсолютная разность (1D)')
ax2.grid(alpha=0.3)

# 3. Динамика интенсивности на экране (QM)
ax3 = fig.add_subplot(gs[1, 0])
I_total_qm = trapezoid(qm_slices_fixed, y, axis=1)
ax3.plot(t_vals, I_total_qm, 'b-', lw=1.5)
ax3.axvline(t_best, color='orange', ls='--', label=f't = {t_best:.3f}')
ax3.set_xlabel('Время $t$')
ax3.set_ylabel('Интегральная интенсивность')
ax3.set_title('Приход пакета на экран')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Полная QM карта (если есть)
ax4 = fig.add_subplot(gs[1, 1])
if 'qm_full_map' in data:
    cax4 = ax4.contourf(x, y, data['qm_full_map'].T, levels=100, cmap='jet')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('QM: $|\\psi(x,y)|^2$')
    fig.colorbar(cax4, ax=ax4, shrink=0.8)
else:
    ax4.text(0.5, 0.5, 'qm_full_map\nне сохранена', 
             transform=ax4.transAxes, ha='center', va='center')
    ax4.set_title('QM карта (отсутствует)')

# 5. Карта расхождения (если есть)
ax5 = fig.add_subplot(gs[1, 2])
if 'diff_full_map' in data:
    cax5 = ax5.contourf(x, y, data['diff_full_map'].T, levels=100, cmap='hot')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title('Разность |QM - Френель|')
    fig.colorbar(cax5, ax=ax5, shrink=0.8)
else:
    ax5.text(0.5, 0.5, 'diff_full_map\nне сохранена', 
             transform=ax5.transAxes, ha='center', va='center')
    ax5.set_title('Карта расхождения (отсутствует)')
# -------------------------------------------------
# График ошибки как функции от x
# -------------------------------------------------
if 'error_vs_x' in data:
    x = data['x_for_error']
    err = data['error_vs_x']

    plt.figure(figsize=(8, 3))
    plt.plot(x, err, 'm-', lw=2)
    plt.xlabel('x')
    plt.ylabel('L2 error')
    plt.title('QM vs Fresnel error along x')
    plt.grid(True)
    plt.xlim(0, x.max())
    plt.show()
    
    
# -------------------------------------------------
# Дополнительно: обновлённая таблица (опционально)
# -------------------------------------------------
# Убрана, чтобы не перегружать; метрики уже в заголовках

# -------------------------------------------------
# Сохранение и отображение
# -------------------------------------------------
plt.suptitle('Сравнение квантовой модели и теории Френеля', fontsize=14, fontweight='bold')
plt.savefig("comparison_analysis.png", dpi=150, bbox_inches='tight')
plt.show()

print("✅ Анализ завершён. График сохранён как 'comparison_analysis.png'")