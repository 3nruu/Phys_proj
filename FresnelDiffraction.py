# fresnel_diffraction.py
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

class FresnelDiffraction:
    """
    Моделирует дифракцию света (или электрона) по теории Френеля.
    Предполагается, что начальное поле U0(x, y) задано непосредственно за апертурой.
    """
    
    def __init__(self, x, y, U0, wavelength):
        """
        Инициализация.
        
        Параметры:
            x : array, координаты по x (размер Nx)
            y : array, координаты по y (размер Ny)
            U0 : 2D array (Ny, Nx) — комплексное поле за апертурой
            wavelength : float — длина волны λ
        """
        self.x = np.array(x)
        self.y = np.array(y)
        self.U0 = np.array(U0, dtype=complex)  # ожидается форма (Ny, Nx)
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength
        
        self.Nx = len(x)
        self.Ny = len(y)
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        
        # Пространственные частоты (для FFT)
        self.fx = fftfreq(self.Nx, d=self.dx)
        self.fy = fftfreq(self.Ny, d=self.dy)
        self.FX, self.FY = np.meshgrid(self.fx, self.fy)
        
        # FFT от начального поля (сохраняем для скорости)
        self.U0_fft = fft2(self.U0)

    def propagate(self, z):
        """
        Вычисляет поле U(x, y, z) на расстоянии z от апертуры.
        
        Параметры:
            z : float — расстояние до плоскости наблюдения
        
        Возвращает:
            U : 2D массив (Ny, Nx) — комплексное поле
        """
        if z == 0:
            return self.U0.copy()
        
        # Передаточная функция Френеля
        H = np.exp(-1j * np.pi * self.wavelength * z * (self.FX**2 + self.FY**2))
        
        # Пропагация через FFT
        Uz_fft = self.U0_fft * H
        Uz = ifft2(Uz_fft)
        
        # Общий фазовый множитель и нормировка
        phase_factor = np.exp(1j * self.k * z) / (1j * self.wavelength * z)
        Uz = phase_factor * Uz
        
        return Uz

    def get_intensity(self, z):
        """Возвращает интенсивность |U(x, y, z)|^2."""
        U = self.propagate(z)
        return np.abs(U)**2

    def get_slice(self, z, x_obs):
        """
        Возвращает 1D срез интенсивности при фиксированном x = x_obs.
        
        Параметры:
            z : расстояние до экрана
            x_obs : координата x на экране
        
        Возвращает:
            y_coords : массив y
            intensity : массив |U(x_obs, y, z)|^2
        """
        intensity_2d = self.get_intensity(z)  # форма (Ny, Nx)
        ix = np.argmin(np.abs(self.x - x_obs))
        return self.y, intensity_2d[:, ix]