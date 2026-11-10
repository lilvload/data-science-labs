import numpy as np
import pandas as pd
import math as mt
import matplotlib.pyplot as plt

def file_parsing(URL, File_name, Data_name):
    d = pd.read_excel(File_name)
    for name, values in d[[Data_name]].items():
        print(values)
        S_real = np.zeros((len(values)))
        for i in range(len(values)):
            S_real[i] = values[i]
    print('Джерело даних: ', URL)
    return S_real

def random_expo(alfa, iter):
    S = np.random.exponential(alfa, iter)
    mS = np.mean(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    print('------- статистичні характеристики ЕКСПОНЕНЦІЙНОГО закону розподілу ВВ -----')
    print('математичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('------------------------------------------------------------------------')
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.title('Гістограма експоненційного закону розподілу')
    plt.show()
    return S

def random_uniform(a, b, iter):
    S = np.random.uniform(a, b, iter)
    mS = np.mean(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    print('------- статистичні характеристики РІВНОМІРНОГО закону розподілу ВВ -----')
    print('математичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('------------------------------------------------------------------------')
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.title('Гістограма рівномірного закону розподілу')
    plt.show()
    return S

def Model_L(n):
    S0 = np.zeros((n))
    for i in range(n):
        S0[i] = (5.0 + 0.005 * i)
    return S0

def Model_Q(n):
    S0 = np.zeros((n))
    for i in range(n):
        S0[i] = (0.0000005 * i * i)
    return S0

def Model_AM(SN, S0N, n):
    SV = np.zeros((n))
    for i in range(n):
        SV[i] = S0N[i] + SN[i]
    return SV

def Model_AV(S0, SV, nAV, Q_AV, iter, dsig):
    SV_AV = np.copy(SV)
    SSAV = np.zeros((nAV))
    SNAV = np.zeros((nAV))
    for i in range(nAV):
        SSAV[i] = mt.ceil(np.random.randint(1, iter))
    SNAV = np.random.normal(0, (Q_AV * dsig), nAV)
    for i in range(nAV):
        k = int(SSAV[i])
        SV_AV[k] = S0[k] + SNAV[i]
    return SV_AV

def MNK_Stat_characteristics(S0):
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):
        Yin[i, 0] = float(S0[i])
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)
    return Yout

def Stat_characteristics(SL, Text):
    Yout = MNK_Stat_characteristics(SL)
    iter = len(SL)
    SL0 = np.zeros((iter))
    for i in range(iter):
        SL0[i] = SL[i] - Yout[i, 0]
    mS = np.mean(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    y_mean = np.mean(SL)
    ss_res = np.sum((SL - Yout[:, 0]) ** 2)
    ss_tot = np.sum((SL - y_mean) ** 2)
    R2 = 1 - ss_res / ss_tot
    print('------------', Text, '-------------')
    print('кількість елементів вибірки =', iter)
    print('математичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('коефіцієнт детермінації R^2 =', R2)
    print('-----------------------------------------------------')
    return

def Plot_AV(S0_L, SV_L, Text):
    plt.clf()
    plt.plot(SV_L)
    plt.plot(S0_L)
    plt.ylabel(Text)
    plt.title(Text)
    plt.show()
    return

if __name__ == '__main__':
    n = 10000
    iter = int(n)
    Q_AV = 3
    nAVv = 10
    nAV = int((iter * nAVv) / 100)

    alfa = 5
    a_u, b_u = -5, 5
    dsig = 5

    SQ = Model_Q(n)
    SC = Model_L(n)
    SExp = random_expo(alfa, iter)
    SUni = random_uniform(a_u, b_u, iter)

    SVE = Model_AM(SExp, SQ, n)
    Plot_AV(SQ, SVE, 'квадратична модель + Експон. шум')
    Stat_characteristics(SVE, 'Вибірка: квадратична + Експон. шум')

    SVU = Model_AM(SUni, SC, n)
    Plot_AV(SC, SVU, 'лінійна модель + Рівном. шум')
    Stat_characteristics(SVU, 'Вибірка: лінійна + Рівном. шум')

    SV_AV = Model_AV(SQ, SVE, nAV, Q_AV, iter, dsig)
    Plot_AV(SQ, SV_AV, 'квадратична модель + Експон. шум + АВ')
    Stat_characteristics(SV_AV, 'Вибірка з АВ: квадратична + Експон. шум')

    SV_AV_U = Model_AV(SC, SVU, nAV, Q_AV, iter, dsig)
    Plot_AV(SC, SV_AV_U, 'лінійна модель + Рівном. шум + АВ')
    Stat_characteristics(SV_AV_U, 'Вибірка з АВ: лінійна + Рівном. шум')

    URL = 'https://www.oschadbank.ua/rates-archive'
    fname = 'Oschadbank (USD).xls'

    SV_buy = file_parsing(URL, fname, 'Купівля')
    Plot_AV(SV_buy, SV_buy, 'Курс USD (Купівля) 2022 - Ощадбанк')
    Stat_characteristics(SV_buy, 'Реальні дані: Купівля USD Ощадбанк')

    SV_sell = file_parsing(URL, fname, 'Продаж')
    Plot_AV(SV_sell, SV_sell, 'Курс USD (Продаж) 2022 - Ощадбанк')
    Stat_characteristics(SV_sell, 'Реальні дані: Продаж USD Ощадбанк')

    SV_nbu = file_parsing(URL, fname, 'КурсНбу')
    Plot_AV(SV_nbu, SV_nbu, 'Курс USD (НБУ) 2022 - Ощадбанк')
    Stat_characteristics(SV_nbu, 'Реальні дані: КурсНбу USD Ощадбанк')
