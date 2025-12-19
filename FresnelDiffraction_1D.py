import numpy as np
from scipy.fft import fft, ifft, fftfreq

class Fresnel1D:
    """
    1D дифракция Френеля–Кирхгофа.
    Продольное направление: z
    Поперечная координата: y
    """

    def __init__(self, y, U0, wavelength):
        """
        y      : 1D массив координат y
        U0(y)  : комплексное поле сразу за апертурой
        """
        self.y = np.array(y)
        self.U0 = np.array(U0, dtype=complex)
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength

        self.N = len(y)
        self.dy = y[1] - y[0]

        # Пространственные частоты
        self.fy = fftfreq(self.N, d=self.dy)

        # FFT начального поля
        self.U0_fft = fft(self.U0)

    def propagate(self, z):
        """
        Пропагация поля на расстояние z
        """
        if z <= 0:
            return self.U0.copy()

        # Передаточная функция Френеля (1D!)
        H = np.exp(-1j * np.pi * self.wavelength * z * self.fy**2)

        Uz_fft = self.U0_fft * H
        Uz = ifft(Uz_fft)

        # Нормировка (КЛЮЧЕВО!)
        Uz *= np.exp(1j * self.k * z) / np.sqrt(1j * self.wavelength * z)

        return Uz

    def intensity(self, z):
        U = self.propagate(z)
        return np.abs(U)**2
